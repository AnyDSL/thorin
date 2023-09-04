#include "thorin/config.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/resolve_loads.h"
#include "thorin/transform/partial_evaluation.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(std::unique_ptr<World>& world)
        : world_(world)
    {}

    World& world() { return *world_; }
    void cleanup();
    void eliminate_tail_rec();
    void eta_conversion();
    void eliminate_params();
    void rebuild();
    void verify_closedness();
    void within(const Def*);
    void clean_pe_infos();

private:
    void cleanup_fix_point();
    void clean_pe_info(std::queue<Continuation*>, Continuation*);
    std::unique_ptr<World>& world_;
    bool todo_ = true;
};

void Cleaner::eliminate_tail_rec() {
    ScopesForest forest(world());
    forest.for_each([&](Scope& scope) {
        auto entry = scope.entry();

        bool only_tail_calls = true;
        bool recursive = false;
        for (auto use : entry->uses()) {
            if (scope.contains(use)) {
                if (use.index() == 0 && use->isa<App>()) {
                    recursive = true;
                    continue;
                } else if (use->isa<Param>())
                    continue; // ignore params

                world().ELOG("non-recursive usage of {} index:{} use:{}", scope.entry()->name(), use.index(), use.def()->to_string());
                only_tail_calls = false;
                break;
            }
        }

        if (recursive && only_tail_calls) {
            auto n = entry->num_params();
            Array<const Def*> args(n);

            for (size_t i = 0; i != n; ++i) {
                args[i] = entry->param(i);

                for (auto use : entry->uses()) {
                    if (scope.contains(use.def())) {
                        auto app = use->isa<App>();
                        if (!app) continue;
                        auto arg = app->arg(i);
                        if (!arg->isa<Bottom>() && arg != args[i]) {
                            args[i] = nullptr;
                            break;
                        }
                    }
                }
            }

            std::vector<const Def*> new_args;

            for (size_t i = 0; i != n; ++i) {
                if (args[i] == nullptr) {
                    new_args.emplace_back(entry->param(i));
                    if (entry->param(i)->order() != 0) {
                        // the resulting function wouldn't be of first order so examine next scope
                        return;
                    }
                }
            }

            if (new_args.size() != n) {
                world().DLOG("tail recursive: {}", entry);
                auto dropped = drop(scope, args);

                entry->jump(dropped, new_args);
                todo_ = true;
                scope.update();
            }
        }
    });
}

void Cleaner::eta_conversion() {
    for (bool todo = true; todo;) {
        todo = false;
        for (auto def : world().copy_defs()) {
            auto continuation = def->isa_nom<Continuation>();
            if (!continuation || !continuation->has_body()) continue;

            // eat calls to known continuations that are only used once
            while (auto callee = continuation->body()->callee()->isa_nom<Continuation>()) {
                auto body = continuation->body();
                if (callee == continuation) break;

                if (callee->has_body() && !world().is_external(callee) && callee->can_be_inlined()) {
                    auto callee_body = callee->body();
                    for (size_t i = 0, e = body->num_args(); i != e; ++i)
                        callee->param(i)->replace_uses(body->arg(i));

                    // because App nodes are hash-consed (thus reusable), there is a risk to invalidate their other uses here, if there are indeed any
                    // can_be_inlined() should account for that by counting reused apps multiple times, but in case it fails we have this pair of asserts as insurance
                    assert(body->num_uses() == 1);
                    continuation->jump(callee_body->callee(), callee_body->args(), callee->debug()); // TODO debug
                    callee->destroy("cleanup: continuation only called once");
                    assert(body->num_uses() == 0);
                    todo_ = todo = true;
                } else
                    break;
            }

            auto body = continuation->body();
            // try to subsume continuations which call a parameter
            // (that is free within that continuation) with that parameter
            if (auto param = body->callee()->isa<Param>()) {
                if (param->continuation() == continuation || world().is_external(continuation))
                    continue;

                if (body->args() == continuation->params_as_defs()) {
                    continuation->replace_uses(body->callee());
                    continuation->destroy("cleanup: calls a parameter (no perm)");
                    todo_ = todo = true;
                    continue;
                }

                // build the permutation of the arguments
                Array<size_t> perm(body->num_args());
                bool is_permutation = true;
                for (size_t i = 0, e = body->num_args(); i != e; ++i)  {
                    auto param_it = std::find(continuation->params().begin(),
                                                continuation->params().end(),
                                                body->arg(i));

                    if (param_it == continuation->params().end()) {
                        is_permutation = false;
                        break;
                    }

                    perm[i] = param_it - continuation->params().begin();
                }

                if (!is_permutation) continue;

                // for every use of the continuation at a call site,
                // permute the arguments and call the parameter instead
                for (auto use : continuation->copy_uses()) {
                    auto uapp = use->isa<App>();
                    if (uapp && use.index() == 0) {
                        for (auto ucontinuation : uapp->using_continuations()) {
                            Array<const Def*> new_args(perm.size());
                            for (size_t i = 0, e = perm.size(); i != e; ++i) {
                                new_args[i] = uapp->arg(perm[i]);
                            }
                            ucontinuation->jump(param, new_args, ucontinuation->debug()); // TODO debug
                            todo_ = todo = true;
                        }
                    }
                }
            }
        }
    }
}

void Cleaner::eliminate_params() {
    // TODO
    for (auto ocontinuation : world().copy_continuations()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (ocontinuation->has_body() && !world().is_external(ocontinuation)) {
            auto obody = ocontinuation->body();
            for (auto use : ocontinuation->uses()) {
                if (use.index() != 0 || !use->isa_nom<Continuation>())
                    goto next_continuation;
            }

            for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
                auto param = ocontinuation->param(i);
                if (param->num_uses() == 0)
                    proxy_idx.push_back(i);
                else
                    param_idx.push_back(i);
            }

            if (!proxy_idx.empty()) {
                auto ncontinuation = world().continuation(
                    world().fn_type(ocontinuation->type()->types().cut(proxy_idx)),
                    ocontinuation->attributes(), ocontinuation->debug_history());
                size_t j = 0;
                for (auto i : param_idx) {
                    ocontinuation->param(i)->replace_uses(ncontinuation->param(j));
                    ncontinuation->param(j++)->set_name(ocontinuation->param(i)->debug_history().name);
                }

                if (!ocontinuation->filter()->is_empty())
                    ncontinuation->set_filter(ocontinuation->filter()->cut(proxy_idx));
                ncontinuation->jump(obody->callee(), obody->args(), ocontinuation->debug());
                ncontinuation->verify();
                ocontinuation->destroy("cleanup: calls a parameter (permutated)");

                for (auto use : ocontinuation->copy_uses()) {
                    auto uapp = use->as<App>();
                    assert(use.index() == 0);
                    for (auto ucontinuation : uapp->using_continuations()) {
                        ucontinuation->jump(ncontinuation, uapp->args().cut(proxy_idx), ucontinuation->debug());
                    }
                }

                todo_ = true;
            }
        }
next_continuation:;
    }
}

void Cleaner::rebuild() {
    auto fresh_world = std::make_unique<World>(world());
    Importer importer(world(), *fresh_world);
    importer.def_old2new_.rehash(world_->defs().capacity());

    for (auto&& [_, def] : world().externals()) {
        if (auto cont = def->isa<Continuation>(); cont && cont->is_exported())
            importer.import(cont);
        if (auto global = def->isa<Global>(); global && global->is_external())
            importer.import(global);
    }

    std::swap(world_, fresh_world);

    // verify(world());

    todo_ |= importer.todo();
}

void Cleaner::verify_closedness() {
    auto check = [&](const Def* def) {
        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert_unused(op->uses().contains(Use(i++, def)) && "can't find def in op's uses");
        }

        for (const auto& use : def->uses()) {
            within(use);
            assert(use->op(use.index()) == def && "use doesn't point to def");
        }
    };

    for (auto def : world().defs())
        check(def);
}

void Cleaner::within(const Def* def) {
    assert(&def->type()->world() == &world());
    assert_unused(world().defs().contains(def));
}

void Cleaner::clean_pe_info(std::queue<Continuation*> queue, Continuation* cur) {
    assert(cur->has_body());
    auto body = cur->body();
    assert(body->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = body->arg(3);
    auto msg = body->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();

    world_->idef(body->callee(), "pe_info was not constant: {}: {}", msg->as_string(), body->arg(2));
    cur->jump(next, {body->arg(0)}, cur->debug()); // TODO debug
    todo_ = true;

    // always re-insert into queue because we've changed cur's jump
    queue.push(cur);
}

void Cleaner::clean_pe_infos() {
    world_->VLOG("cleaning remaining pe_infos");
    std::queue<Continuation*> queue;
    ContinuationSet done;
    auto enqueue = [&](Continuation* continuation) {
        if (done.emplace(continuation).second)
            queue.push(continuation);
    };

    for (auto&& [_, def] : world().externals())
        if (auto cont = def->isa<Continuation>(); cont && cont->has_body()) enqueue(cont);

    while (!queue.empty()) {
        auto continuation = pop(queue);

        if (continuation->has_body()) {
            if (auto body = continuation->body()->isa<App>()) {
                if (auto callee = body->callee()->isa_nom<Continuation>(); callee && callee->intrinsic() == Intrinsic::PeInfo) {
                    clean_pe_info(queue, continuation);
                    continue;
                }
            }
        }

        for (auto succ : continuation->succs())
            enqueue(succ);
    }
}

void Cleaner::cleanup_fix_point() {
    int i = 0;
    for (; todo_; ++i) {
        world_->VLOG("iteration: {}", i);
        todo_ = false;
        if (world_->is_pe_done())
            eliminate_tail_rec();
        eta_conversion();
        eliminate_params();
        rebuild(); // resolve replaced defs before going to resolve_loads
        todo_ |= resolve_loads(world());
        rebuild();
        if (!world().is_pe_done())
            todo_ |= partial_evaluation(*world_);
        else
            clean_pe_infos();
    }
}

void Cleaner::cleanup() {
    world_->VLOG("start cleanup");
    cleanup_fix_point();

    if (!world().is_pe_done()) {
        world().mark_pe_done();
        for (auto def : world().defs()) {
            if (auto cont = def->isa_nom<Continuation>())
                cont->destroy_filter();
        }

        todo_ = true;
        cleanup_fix_point();
    }

    world_->VLOG("end cleanup");
#if THORIN_ENABLE_CHECKS
    verify_closedness();
    debug_verify(world());
#endif
}

void cleanup_world(std::unique_ptr<World>& world) { Cleaner(world).cleanup(); }

}
