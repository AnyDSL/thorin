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
    Cleaner(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
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
    World& world_;
    bool todo_ = true;
};

void Cleaner::eliminate_tail_rec() {
    Scope::for_each(world_, [&](Scope& scope) {
        auto entry = scope.entry();

        bool only_tail_calls = true;
        bool recursive = false;
        for (auto use : entry->uses()) {
            if (scope.contains(use)) {
                if (use.index() != 0 || !use->isa<App>()) {
                    only_tail_calls = false;
                    break;
                } else {
                    recursive = true;
                }
            }
        }

        if (recursive && only_tail_calls) {
            auto n = entry->num_params();
            Array<const Def*> args(n);

            for (size_t i = 0; i != n; ++i) {
                args[i] = entry->param(i);

                for (auto use : entry->uses()) {
                    if (scope.contains(use)) {
                        auto arg = use->as<App>()->arg(i);
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
        for (auto continuation : world().continuations()) {
            if (!continuation->has_body()) continue;
            auto body = continuation->body();

            // eat calls to known continuations that are only used once
            while (auto callee = body->callee()->isa_continuation()) {
                if (callee == continuation) break;

                if (callee->num_uses_excluding_params() == 1 && callee->has_body() && !callee->is_exported()) {
                    auto callee_body = callee->body();
                    for (size_t i = 0, e = body->num_args(); i != e; ++i)
                        callee->param(i)->replace(body->arg(i));
                    continuation->jump(callee_body->callee(), callee_body->args(), callee->debug()); // TODO debug
                    callee->destroy();
                    todo_ = todo = true;
                } else
                    break;
            }

            // try to subsume continuations which call a parameter
            // (that is free within that continuation) with that parameter
            if (auto param = body->callee()->isa<Param>()) {
                if (param->continuation() == continuation || continuation->is_exported())
                    continue;

                if (body->args() == continuation->params_as_defs()) {
                    continuation->replace(body->callee());
                    continuation->destroy();
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
    for (auto ocontinuation : world().copy_continuations()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (ocontinuation->has_body() && !ocontinuation->is_exported()) {
            auto obody = ocontinuation->body();
            for (auto use : ocontinuation->uses()) {
                if (use.index() != 0 || !use->isa_continuation())
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
                    world().fn_type(ocontinuation->type()->ops().cut(proxy_idx)),
                    ocontinuation->attributes(), ocontinuation->debug_history());
                size_t j = 0;
                for (auto i : param_idx) {
                    ocontinuation->param(i)->replace(ncontinuation->param(j));
                    ncontinuation->param(j++)->set_name(ocontinuation->param(i)->debug_history().name);
                }

                if (!ocontinuation->filter()->is_empty())
                    ncontinuation->set_filter(ocontinuation->filter()->cut(proxy_idx));
                ncontinuation->jump(obody->callee(), obody->args(), ocontinuation->debug());
                ncontinuation->verify();
                ocontinuation->destroy();

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
    verify(world());

    Importer importer(world_);
    importer.type_old2new_.rehash(world_.types().capacity());
    importer.def_old2new_.rehash(world_.primops().capacity());

    for (auto continuation : world().exported_continuations())
        importer.import(continuation);

    swap(importer.world(), world_);
    todo_ |= importer.todo();
}

void Cleaner::verify_closedness() {
    auto check = [&](const Def* def) {
        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert_unused(op->uses_.contains(Use(i++, def)) && "can't find def in op's uses");
        }

        for (const auto& use : def->uses_) {
            within(use);
            assert(use->op(use.index()) == def && "use doesn't point to def");
        }
    };

    for (auto primop : world().primops())
        check(primop);
    for (auto continuation : world().continuations()) {
        check(continuation);
        for (auto param : continuation->params())
            check(param);
    }
}

void Cleaner::within(const Def* def) {
    assert(world().types().contains(def->type()));
    if (auto primop = def->isa<PrimOp>())
        assert_unused(world().primops().contains(primop));
    else if (auto continuation = def->isa_continuation())
        assert_unused(world().continuations().contains(continuation));
    else if (def->isa<App>() || def->isa<Filter>())
        {}
    else
        within(def->as<Param>()->continuation());
}

void Cleaner::clean_pe_info(std::queue<Continuation*> queue, Continuation* cur) {
    assert(cur->has_body());
    auto body = cur->body();
    assert(body->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = body->arg(3);
    auto msg = body->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();

    world_.idef(body->callee(), "pe_info was not constant: {}: {}", msg->as_string(), body->arg(2));
    cur->jump(next, {body->arg(0)}, cur->debug()); // TODO debug
    todo_ = true;

    // always re-insert into queue because we've changed cur's jump
    queue.push(cur);
}

void Cleaner::clean_pe_infos() {
    world_.VLOG("cleaning remaining pe_infos");
    std::queue<Continuation*> queue;
    ContinuationSet done;
    auto enqueue = [&](Continuation* continuation) {
        if (done.emplace(continuation).second)
            queue.push(continuation);
    };

    for (auto continuation : world().exported_continuations())
        enqueue(continuation);

    while (!queue.empty()) {
        auto continuation = pop(queue);

        if (continuation->has_body()) {
            if (auto body = continuation->body()->isa<App>()) {
                if (auto callee = body->callee()->isa_continuation(); callee && callee->intrinsic() == Intrinsic::PeInfo) {
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
        world_.VLOG("iteration: {}", i);
        verify(world());
        todo_ = false;
        if (world_.is_pe_done())
            eliminate_tail_rec();
        rebuild();
        eta_conversion();
        verify(world());
        eliminate_params();
        rebuild(); // resolve replaced defs before going to resolve_loads
        todo_ |= resolve_loads(world());
        rebuild();
        if (!world().is_pe_done())
            todo_ |= partial_evaluation(world_);
        else
            clean_pe_infos();
    }
}

void Cleaner::cleanup() {
    world_.VLOG("start cleanup");
    cleanup_fix_point();

    if (!world().is_pe_done()) {
        world().mark_pe_done();
        for (auto continuation : world().continuations())
            continuation->destroy_filter();
        todo_ = true;
        cleanup_fix_point();
    }

    world_.VLOG("end cleanup");
#if THORIN_ENABLE_CHECKS
    verify_closedness();
    debug_verify(world());
#endif
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
