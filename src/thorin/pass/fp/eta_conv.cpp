#include "thorin/pass/fp/eta_conv.h"

#include "thorin/analyses/scope.h"

namespace thorin {

/// For now, only eta-convert <code>lam x.e x</code> if e is a @p Var or a @p Lam.
static const App* eta_rule(Lam* lam) {
    if (auto app = lam->body()->isa<App>()) {
        if (app->arg() == lam->var() && (app->callee()->isa<Var>() || app->callee()->isa<Lam>()))
            return app;
    }
    return nullptr;
}

const Def* EtaConv::rewrite(const Def* def) {
    if (def->isa<Var>() || def->isa<Proxy>()) return def;

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
            if (auto app = eta_rule(lam); app && !irreducible_.contains(lam)) {
                put<Reduce>(lam);
                auto new_def = def->refine(i, app->callee());
                world().DLOG("eta-reduction '{}' -> '{}' by eliminating '{}'", def, new_def, lam);
                return new_def;
            }

            if (!is_callee(def, i) && expand_.contains(lam)) {
                auto [j, ins] = def2exp_.emplace(def, nullptr);
                if (ins) {
                    auto wrap = lam->stub(world(), lam->type(), lam->dbg());
                    wrap->set_name(std::string("eta_") + lam->debug().name);
                    wrap->app(lam, wrap->var());
                    irreducible_.emplace(wrap);
                    j->second = def->refine(i, wrap);
                    world().DLOG("eta-expansion '{}' -> '{}' using '{}'", def, j->second, wrap);
                }

                return j->second;
            }
        }
    }

    return def;
}

undo_t EtaConv::analyze(const Def* def) {
    if (auto var = def->isa<Var>()) {
        if (auto lam = var->nom()->isa_nom<Lam>()) {
            auto [undo, ins] = put<Reduce>(lam);
            auto succ = irreducible_.emplace(lam).second;
            if (!ins && succ) {
                world().DLOG("irreducible: {}; found {}", lam, var);
                return undo;
            }
        }
    }

    auto cur_lam = descend<Lam>(def);
    if (cur_lam == nullptr) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
            if (expand_.contains(lam)) continue;

            if (is_callee(def, i)) {
                callee_.emplace(lam).second;
                auto u = contains<Non_Callee_1>(lam);
                bool non_callee = u != No_Undo;

                if (non_callee) {
                    world().DLOG("Callee -> Expand: '{}'", lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, u);
                } else {
                    world().DLOG("Irreducible/Callee -> Callee: '{}'", lam);
                }
            } else {
                bool callee = callee_.contains(lam);
                auto [u, first_non_callee] = put<Non_Callee_1>(lam);

                if (callee) {
                    world().DLOG("Callee -> Expand: '{}'", lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, u);
                } else {
                    if (first_non_callee) {
                        world().DLOG("Irreducible -> Non_Callee_1: '{}'", lam);
                    } else {
                        world().DLOG("Non_Callee_1 -> Expand: '{}'", lam);
                        expand_.emplace(lam);
                        undo = std::min(undo, u);
                    }
                }
            }
        }
    }

    return undo;
}

}
