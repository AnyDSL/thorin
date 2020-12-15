#include "thorin/config.h"
#include "thorin/world.h"
#include "thorin/rewrite.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/partial_evaluation.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    void run();
    void eta_conversion();
    void eliminate_vars();
    void verify_closedness();
    void within(const Def*);
    void clean_pe_infos();

private:
    void cleanup_fix_point();
    World& world_;
    bool todo_ = true;
};

// TODO remove
bool is_var(const Def* def) {
    if (def->isa<Var>()) return true;
    if (auto extract = def->isa<Extract>()) return extract->tuple()->isa<Var>();
    return false;
}

void Cleaner::eta_conversion() {
    for (bool todo = true; todo;) {
        todo = false;

        for (auto lam : world().copy_lams()) {
            if (!lam->is_set()) continue;

            // eat calls to known lams that are only used once
            while (true) {
                if (auto app = lam->body()->isa<App>()) {
                    if (auto callee = app->callee()->isa_nom<Lam>()) {
                        if (!callee->is_set() || callee->is_external() || callee->num_uses() > 2) break;
                        bool ok = true;
                        for (auto use : callee->uses()) { // 2 iterations at max - see above
                            if (!use->isa<App>() && !use->isa<Var>())
                                ok = false;
                        }
                        if (!ok) break;

                        callee->var()->replace(app->arg());
                        lam->set_body(callee->body());
                        callee->unset();
                        todo_ = todo = true;
                        continue;
                    }
                }
                break;
            }

            auto app = lam->body()->isa<App>();
            if (!app) continue;

            // try to subsume lams which call a var
            // (that is free within that lam) with that var
            auto callee = app->callee();
            if (is_var(callee)) {
                if (get_var_lam(callee) == lam || lam->is_external())
                    continue;

                if (app->arg() == lam->var()) {
                    lam->replace(callee);
                    lam->unset();
                    todo_ = todo = true;
                    continue;
                }

#if 0
                // build the permutation of the arguments
                Array<size_t> perm(app->num_args());
                bool is_permutation = true;
                auto vars = lam->vars();
                for (size_t i = 0, e = app->num_args(); i != e; ++i)  {
                    auto var_it = std::find(vars.begin(), vars.end(), app->arg(i));

                    if (var_it == vars.end()) {
                        is_permutation = false;
                        break;
                    }

                    perm[i] = var_it - vars.begin();
                }

                if (!is_permutation) continue;

                // for every use of the lam at an pp,
                // permute the arguments and call the var instead
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

void Cleaner::eliminate_vars() {
    for (auto old_lam : world().copy_lams()) {
        if (!old_lam->is_set()) continue;

        std::vector<size_t> proxy_idx; // indices of vars we eliminate
        std::vector<size_t> var_idx; // indices of vars we keep

        auto old_app = old_lam->body()->isa<App>();
        if (old_app == nullptr || world().is_external(old_lam)) continue;

        for (auto use : old_lam->uses()) {
            if (use->isa<Var>()) continue; // ignore old_lam's Var
            if (use.index() != 0 || !use->isa<App>())
                goto next_lam;
        }

        // maybe the whole var tuple is passed somewhere?
        if (old_lam->num_vars() != 1) {
            for (auto use : old_lam->var()->uses()) {
                if (!use->isa<Extract>())
                    goto next_lam;
            }
        }

        for (size_t i = 0, e = old_lam->num_vars(); i != e; ++i) {
            auto var = old_lam->var(i);
            if (var->num_uses() == 0)
                proxy_idx.push_back(i);
            else
                var_idx.push_back(i);
        }

        if (!proxy_idx.empty()) {
            auto old_dom = old_lam->type()->dom();

            const Def* new_dom;
            if (auto sigma = old_dom->isa<Sigma>())
                new_dom = world().sigma(sigma->ops().cut(proxy_idx));
            else {
                assert(proxy_idx.size() == 1 && proxy_idx[0] == 0);
                new_dom = world().sigma();
            }

            auto cn = world().cn(new_dom);
            auto new_lam = world().nom_lam(cn, old_lam->cc(), old_lam->debug_history());
            size_t j = 0;
            for (auto i : var_idx) {
                old_lam->var(i)->replace(new_lam->var(j));
                new_lam->var(j++, old_lam->var(i)->debug_history());
            }

            new_lam->set_filter(old_lam->filter());
            new_lam->app(old_app->callee(), old_app->args(), old_app->dbg());
            old_lam->unset();

            for (auto use : old_lam->copy_uses()) {
                if (use->isa<Var>()) continue; // ignore old_lam's Var
                auto use_app = use->as<App>();
                assert(use.index() == 0);
                use_app->replace(world().app(new_lam, use_app->args().cut(proxy_idx), use_app->dbg()));
            }

            todo_ = true;
        }
next_lam:;
    }
}

void Cleaner::verify_closedness() {
    for (auto def : world().defs()) {
        if (!def->is_set()) continue;

        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            within(def->op(i));
            assert((def->op(i)->is_const() || def->op(i)->uses_.contains(Use(def, i))) && "can't find def in op's uses");
        }

        for (const auto& use : def->uses_) {
            within(use);
            assert((use.is_used_as_type() || use->op(use.index()) == def) && "use doesn't point to def");
        }
    }
}

void Cleaner::within(const Def* def) {
    assert_unused(world().defs().contains(def));
    assert_unused(world().defs().contains(def->type()));
}

void Cleaner::clean_pe_infos() {
#if 0
    world().rewrite("cleaning remaining pe_infos",
        [&](const Scope& scope) {
            return scope.entry()->isa<Lam>();
        },
        [&](const Def* old_def) -> const Def* {
            if (auto app = old_def->isa<App>()) {
                if (auto callee = app->callee()->isa_nom<Lam>()) {
                    if (callee->intrinsic() == Lam::Intrinsic::PeInfo) {
                        auto next = app->arg(3);
                        assert(app->arg(2)->is_const());
                        world().idef(app->callee(), "pe_info not constant: {}: {}", "TODO", app->arg(2));
                        return world().app(next, {app->arg(0)}, app->debug());
                    }
                }
            }
            return nullptr;
        });
#endif
}

void Cleaner::cleanup_fix_point() {
    int i = 0;
    for (; todo_; ++i) {
        world().VLOG("iteration: {}", i);
        todo_ = false;
        eta_conversion();
        eliminate_vars();
        cleanup(world_); // resolve replaced defs before going to resolve_loads
        if (!world().is_pe_done())
            todo_ |= partial_evaluation(world_);
        else
            clean_pe_infos();
    }
}

void Cleaner::run() {
    world().VLOG("start cleanup");
    cleanup_fix_point();

    if (!world().is_pe_done()) {
        world().mark_pe_done();
        //for (auto lam : world().lams())
            //lam->destroy_filter();
        todo_ = true;
        cleanup_fix_point();
    }

    world().VLOG("end cleanup");
#if THORIN_ENABLE_CHECKS
    verify_closedness();
#endif
}

void cleanup_world(World& world) { Cleaner(world).run(); }

}
