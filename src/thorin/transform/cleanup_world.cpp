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
#include "thorin/util/log.h"

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
            if (use->isa<Param>()) continue;
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
                    if (use->isa<Param>()) continue;
                    if (scope.contains(use)) {
                        auto arg = use->as<App>()->arg(i);
                        if (!arg->isa<Bottom>() && arg != args[i]) {
                            args[i] = world().top(arg->type());
                            break;
                        }
                    }
                }
            }

            std::vector<const Def*> new_args;

            for (size_t i = 0; i != n; ++i) {
                if (args[i]->isa<Top>()) {
                    new_args.emplace_back(entry->param(i));
                    if (entry->param(i)->type()->order() != 0) {
                        // the resulting function wouldn't be of first order so examine next scope
                        return;
                    }
                }
            }

            if (new_args.size() != n) {
                DLOG("tail recursive: {}", entry);
                auto dropped = drop(scope, args);

                entry->app(dropped, new_args);
                todo_ = true;
                scope.update();
            }
        }
    });
}

void Cleaner::eta_conversion() {
#if 0
    for (bool todo = true; todo;) {
        todo = false;

        for (auto def : world().defs()) {
            auto lam = def->isa_lam();
            if (lam == nullptr) continue;

            // eat calls to known lams that are only used once
            while (auto callee = lam->app()->callee()->isa_lam()) {
                if (callee->num_uses() == 1 && !callee->is_empty() && !callee->is_external()) {
                    for (size_t i = 0, e = lam->num_args(); i != e; ++i)
                        callee->param(i)->replace(lam->arg(i));
                    lam->jump(callee->callee(), callee->args(), callee->jump_debug());
                    callee->destroy();
                    todo_ = todo = true;
                } else
                    break;
            }

            // try to subsume lams which call a parameter
            // (that is free within that lam) with that parameter
            if (auto param = lam->callee()->isa<Param>()) {
                if (param->lam() == lam || lam->is_external())
                    continue;

                if (lam->arg() == lam->param()) {
                    lam->replace(lam->callee());
                    lam->destroy();
                    todo_ = todo = true;
                    continue;
                }

                // build the permutation of the arguments
                Array<size_t> perm(lam->num_args());
                bool is_permutation = true;
                for (size_t i = 0, e = lam->num_args(); i != e; ++i)  {
                    auto param_it = std::find(lam->params().begin(),
                                                lam->params().end(),
                                                lam->arg(i));

                    if (param_it == lam->params().end()) {
                        is_permutation = false;
                        break;
                    }

                    perm[i] = param_it - lam->params().begin();
                }

                if (!is_permutation) continue;

                // for every use of the lam at a call site,
                // permute the arguments and call the parameter instead
                for (auto use : lam->copy_uses()) {
                    auto ulam = use->isa_lam();
                    if (ulam && use.index() == 0) {
                        Array<const Def*> new_args(perm.size());
                        for (size_t i = 0, e = perm.size(); i != e; ++i) {
                            new_args[i] = ulam->arg(perm[i]);
                        }
                        ulam->jump(param, new_args, ulam->jump_debug());
                        todo_ = todo = true;
                    }
                }
            }
        }
    }
#endif
}

void Cleaner::eliminate_params() {
    for (auto old_lam : world().copy_lams()) {
        std::vector<size_t> proxy_idx; // indices of params we eliminate
        std::vector<size_t> param_idx; // indices of params we keep

        auto old_app = old_lam->app();
        if (old_app == nullptr || world().is_external(old_lam)) continue;

        for (auto use : old_lam->uses()) {
            if (use->isa<Param>()) continue; // ignore old_lam's Param
            if (use.index() != 0 || !use->isa<App>())
                goto next_lam;
        }

        // maybe the whole param tuple is passed somewhere?
        if (old_lam->num_params() != 1) {
            for (auto use : old_lam->param()->uses()) {
                if (!use->isa<Extract>())
                    goto next_lam;
            }
        }

        for (size_t i = 0, e = old_lam->num_params(); i != e; ++i) {
            auto param = old_lam->param(i);
            if (param->num_uses() == 0)
                proxy_idx.push_back(i);
            else
                param_idx.push_back(i);
        }

        if (!proxy_idx.empty()) {
            auto old_domain = old_lam->type()->domain();

            const Def* new_domain;
            if (auto sigma = old_domain->isa<Sigma>())
                new_domain = world().sigma(sigma->ops().cut(proxy_idx));
            else {
                assert(proxy_idx.size() == 1 && proxy_idx[0] == 0);
                new_domain = world().sigma();
            }

            auto cn = world().cn(new_domain);
            auto new_lam = world().lam(cn, old_lam->cc(), old_lam->intrinsic(), old_lam->debug_history());
            size_t j = 0;
            for (auto i : param_idx) {
                old_lam->param(i)->replace(new_lam->param(j));
                new_lam->param(j++)->debug() = old_lam->param(i)->debug_history();
            }

            Array<const Def*> new_filter(param_idx.size());
            size_t i = 0;
            for (auto j : param_idx)
                new_filter[i++] = old_lam->filter(j);

            new_lam->set_filter(new_filter);
            new_lam->app(old_app->callee(), old_app->args(), old_app->debug());
            old_lam->destroy();

            for (auto use : old_lam->copy_uses()) {
                if (use->isa<Param>()) continue; // ignore old_lam's Param
                auto use_app = use->as<App>();
                assert(use.index() == 0);
                use_app->replace(world().app(new_lam, use_app->args().cut(proxy_idx), use_app->debug()));
            }

            todo_ = true;
        }
next_lam:;
    }
}

void Cleaner::rebuild() {
    Importer importer(world_);
    importer.old2new_.rehash(world_.defs().capacity());

#if THORIN_ENABLE_CHECKS
    world_.swap_breakpoints(importer.world());
#endif

    for (auto external : world().externals())
        importer.import(external);

    swap(importer.world(), world_);
    todo_ |= importer.todo();
}

void Cleaner::verify_closedness() {
    for (auto def : world().defs()) {
        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert_unused(op->uses_.contains(Use(i++, def)) && "can't find def in op's uses");
        }

        for (const auto& use : def->uses_) {
            within(use);
            assert(use->op(use.index()) == def && "use doesn't point to def");
        }
    }
}

void Cleaner::within(const Def* def) {
    assert_unused(world().defs().contains(def));
    assert_unused(world().defs().contains(def->type()));
}

void Cleaner::clean_pe_info(std::queue<Lam*> queue, Lam* cur) {
    auto app = cur->app();
    assert(app->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = app->arg(3);
    auto msg = app->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();

    assert(!is_const(app->arg(2)));
    IDEF(app->callee(), "pe_info not constant: {}: {}", msg->as_string(), app->arg(2));
    cur->app(next, {app->arg(0)}, app->debug());
    todo_ = true;

    // always re-insert into queue because we've changed cur's jump
    queue.push(cur);
}

void Cleaner::clean_pe_infos() {
    VLOG("cleaning remaining pe_infos");
    std::queue<Lam*> queue;
    LamSet done;
    auto enqueue = [&](Lam* lam) {
        if (done.emplace(lam).second)
            queue.push(lam);
    };
    for (auto external : world().externals()) {
        enqueue(external);
    }

    while (!queue.empty()) {
        auto lam = pop(queue);

        if (auto app = lam->app()) {
            if (auto callee = app->callee()->isa_lam()) {
                if (callee->intrinsic() == Intrinsic::PeInfo) {
                    clean_pe_info(queue, lam);
                    continue;
                }
            }
        }

        for (auto succ : lam->succs())
            enqueue(succ);
    }
}

void Cleaner::cleanup_fix_point() {
    int i = 0;
    for (; todo_; ++i) {
        VLOG("iteration: {}", i);
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
    VLOG("start cleanup");
    cleanup_fix_point();

    if (!world().is_pe_done()) {
        world().mark_pe_done();
        //for (auto lam : world().lams())
            //lam->destroy_filter();
        todo_ = true;
        cleanup_fix_point();
    }

    VLOG("end cleanup");
#if THORIN_ENABLE_CHECKS
    verify_closedness();
    debug_verify(world());
#endif
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
