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
    void clean_pe_info(std::queue<Lam*>, Lam*);
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
        for (auto def : world().copy_defs()) {
            auto lambda = def->isa_nom<Lam>();
            if (!lambda || !lambda->has_body()) continue;
            auto body = lambda->body();

            // eat calls to known lambdas that are only used once
            while (auto callee = body->callee()->isa_nom<Lam>()) {
                if (callee == lambda) break;

                if (callee->can_be_inlined() && callee->has_body() && !world().is_external(callee)) {
                    auto callee_body = callee->body();
                    for (size_t i = 0, e = body->num_args(); i != e; ++i)
                        callee->param(i)->replace_uses(body->arg(i));

                    // because App nodes are hash-consed (thus reusable), there is a risk to invalidate their other uses here, if there are indeed any
                    // can_be_inlined() should account for that by counting reused apps multiple times, but in case it fails we have this pair of asserts as insurance
                    assert(body->num_uses() == 1);
                    lambda->jump(callee_body->callee(), callee_body->args(), callee->debug()); // TODO debug
                    callee->destroy("cleanup: lambda only called once");
                    assert(body->num_uses() == 0);
                    todo_ = todo = true;
                } else
                    break;
            }

            // try to subsume lamdas which call a parameter
            // (that is free within that lambda's body) with that parameter
            if (auto param = body->callee()->isa<Param>()) {
                if (param->lambda() == lambda || world().is_external(lambda))
                    continue;

                if (body->args() == lambda->params_as_defs()) {
                    lambda->replace_uses(body->callee());
                    lambda->destroy("cleanup: calls a parameter (no perm)");
                    todo_ = todo = true;
                    continue;
                }

                // build the permutation of the arguments
                Array<size_t> perm(body->num_args());
                bool is_permutation = true;
                for (size_t i = 0, e = body->num_args(); i != e; ++i)  {
                    auto param_it = std::find(lambda->params().begin(),
                                                lambda->params().end(),
                                                body->arg(i));

                    if (param_it == lambda->params().end()) {
                        is_permutation = false;
                        break;
                    }

                    perm[i] = param_it - lambda->params().begin();
                }

                if (!is_permutation) continue;

                // for every use of the lambda at a call site,
                // permute the arguments and call the parameter instead
                for (auto use : lambda->copy_uses()) {
                    auto uapp = use->isa<App>();
                    if (uapp && use.index() == 0) {
                        for (auto user : uapp->using_lambdas()) {
                            Array<const Def*> new_args(perm.size());
                            for (size_t i = 0, e = perm.size(); i != e; ++i) {
                                new_args[i] = uapp->arg(perm[i]);
                            }
                            user->jump(param, new_args, user->debug()); // TODO debug
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
    for (auto olam : world().copy_lams()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (olam->has_body() && !world().is_external(olam)) {
            auto obody = olam->body();
            for (auto use : olam->uses()) {
                if (use.index() != 0 || !use->isa_nom<Lam>())
                    goto next_lam;
            }

            for (size_t i = 0, e = olam->num_params(); i != e; ++i) {
                auto param = olam->param(i);
                if (param->num_uses() == 0)
                    proxy_idx.push_back(i);
                else
                    param_idx.push_back(i);
            }

            if (!proxy_idx.empty()) {
                auto nlam = world().lambda(
                    world().fn_type(olam->type()->ops().cut(proxy_idx)),
                    olam->attributes(), olam->debug_history());
                size_t j = 0;
                for (auto i : param_idx) {
                    olam->param(i)->replace_uses(nlam->param(j));
                    nlam->param(j++)->set_name(olam->param(i)->debug_history().name);
                }

                if (!olam->filter()->is_empty())
                    nlam->set_filter(olam->filter()->cut(proxy_idx));
                nlam->jump(obody->callee(), obody->args(), olam->debug());
                nlam->verify();
                olam->destroy("cleanup: calls a parameter (permutated)");

                for (auto use : olam->copy_uses()) {
                    auto uapp = use->as<App>();
                    assert(use.index() == 0);
                    for (auto user : uapp->using_lambdas()) {
                        user->jump(nlam, uapp->args().cut(proxy_idx), user->debug());
                    }
                }

                todo_ = true;
            }
        }
next_lam:;
    }
}

void Cleaner::rebuild() {
    Importer importer(world_);
    importer.type_old2new_.rehash(world_.types().capacity());
    importer.def_old2new_.rehash(world_.defs().capacity());

    for (auto&& [_, lam] : world().externals()) {
        if (lam->is_exported())
            importer.import(lam);
    }

    swap(importer.world(), world_);

    verify(world());

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

    for (auto def : world().defs())
        check(def);
}

void Cleaner::within(const Def* def) {
    if (def->isa<Param>()) return; // TODO remove once Params are within World's sea of nodes
    assert(world().types().contains(def->type()));
    assert_unused(world().defs().contains(def));
}

void Cleaner::clean_pe_info(std::queue<Lam*> queue, Lam* cur) {
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
    std::queue<Lam*> queue;
    LamSet done;
    auto enqueue = [&](Lam* lam) {
        if (done.emplace(lam).second)
            queue.push(lam);
    };

    for (auto&& [_, lam] : world().externals())
        if (lam->has_body()) enqueue(lam);

    while (!queue.empty()) {
        auto lambda = pop(queue);

        if (lambda->has_body()) {
            if (auto body = lambda->body()->isa<App>()) {
                if (auto callee = body->callee()->isa_nom<Lam>(); callee && callee->intrinsic() == Intrinsic::PeInfo) {
                    clean_pe_info(queue, lambda);
                    continue;
                }
            }
        }

        for (auto succ : lambda->succs())
            enqueue(succ);
    }
}

void Cleaner::cleanup_fix_point() {
    int i = 0;
    for (; todo_; ++i) {
        world_.VLOG("iteration: {}", i);
        todo_ = false;
        if (world_.is_pe_done())
            eliminate_tail_rec();
        eta_conversion();
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
        for (auto def : world().defs()) {
            if (auto lam = def->isa_nom<Lam>())
                lam->destroy_filter();
        }

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
