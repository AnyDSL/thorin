#include "thorin/config.h"
#include "thorin/util.h"
#include "thorin/world.h"
#include "thorin/rewrite.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
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
    void run();
    void eliminate_tail_rec();
    void eta_conversion();
    void eliminate_params();
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
                        if (!arg->isa<Bot>() && arg != args[i]) {
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

// TODO remove
bool is_param(const Def* def) {
    if (def->isa<Param>()) return true;
    if (auto extract = def->isa<Extract>()) return extract->agg()->isa<Param>();
    return false;
}

void Cleaner::eta_conversion() {
    for (bool todo = true; todo;) {
        todo = false;

        for (auto lam : world().copy_lams()) {
            if (!lam->is_set()) continue;

            // eat calls to known lams that are only used once
            while (true) {
                if (auto app = lam->app()) {
                    if (auto callee = app->callee()->isa_nominal<Lam>()) {
                        if (!callee->is_set() || callee->is_external() || callee->num_uses() > 2) break;
                        bool ok = true;
                        for (auto use : callee->uses()) { // 2 iterations at max - see above
                            if (!use->isa<App>() && !use->isa<Param>())
                                ok = false;
                        }
                        if (!ok) break;

                        callee->param()->replace(app->arg());
                        lam->set_body(callee->body());
                        callee->unset();
                        todo_ = todo = true;
                        continue;
                    }
                }
                break;
            }

            auto app = lam->app();
            if (!app) continue;

            // try to subsume lams which call a parameter
            // (that is free within that lam) with that parameter
            auto callee = app->callee();
            if (is_param(callee)) {
                if (get_param_lam(callee) == lam || lam->is_external())
                    continue;

                if (app->arg() == lam->param()) {
                    lam->replace(callee);
                    lam->unset();
                    todo_ = todo = true;
                    continue;
                }

#if 0
                // build the permutation of the arguments
                Array<size_t> perm(app->num_args());
                bool is_permutation = true;
                auto params = lam->params();
                for (size_t i = 0, e = app->num_args(); i != e; ++i)  {
                    auto param_it = std::find(params.begin(), params.end(), app->arg(i));

                    if (param_it == params.end()) {
                        is_permutation = false;
                        break;
                    }

                    perm[i] = param_it - params.begin();
                }

                if (!is_permutation) continue;

                // for every use of the lam at an pp,
                // permute the arguments and call the parameter instead
                for (auto use : lam->copy_uses()) {
                    auto uapp = use->isa<App>();
                    if (uapp && use.index() == 0) {
                        Array<const Def*> new_args(perm.size());
                        for (size_t i = 0, e = perm.size(); i != e; ++i) {
                            new_args[i] = uapp->arg(perm[i]);
                        }
                        uapp->replace(world().app(callee, new_args, uapp->debug()));
                        todo_ = todo = true;
                    }
                }
#endif
            }
        }
    }
}

void Cleaner::eliminate_params() {
    for (auto old_lam : world().copy_lams()) {
        if (!old_lam->is_set()) continue;

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
                new_lam->param(j++, old_lam->param(i)->debug_history());
            }

            new_lam->set_filter(old_lam->filter());
            new_lam->app(old_app->callee(), old_app->args(), old_app->debug());
            old_lam->unset();

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

void Cleaner::verify_closedness() {
    for (auto def : world().defs()) {
        if (!def->is_set()) continue;

        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert_unused(op->uses_.contains(Use(def, i++)) && "can't find def in op's uses");
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
    assert(app->arg(1)->type() == world().ptr_type(world().unsafe_variadic(world().type_pu8())));
    auto next = app->arg(3);
    //auto msg = app->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<Variadic>();

    assert(!is_const(app->arg(2)));
    IDEF(app->callee(), "pe_info not constant: {}: {}", "TODO", app->arg(2));
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
        if (lam->is_set() && done.emplace(lam).second)
            queue.push(lam);
    };
    for (auto external : world().externals()) {
        if (auto lam = external->isa<Lam>())
            enqueue(lam);
    }

    while (!queue.empty()) {
        auto lam = pop(queue);

        if (auto app = lam->app()) {
            if (auto callee = app->callee()->isa_nominal<Lam>()) {
                if (callee->intrinsic() == Lam::Intrinsic::PeInfo) {
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
        cleanup(world_); // resolve replaced defs before going to resolve_loads
        todo_ |= resolve_loads(world());
        cleanup(world_);
        if (!world().is_pe_done())
            todo_ |= partial_evaluation(world_);
        else
            clean_pe_infos();
    }
}

void Cleaner::run() {
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

void cleanup_world(World& world) { Cleaner(world).run(); }

}
