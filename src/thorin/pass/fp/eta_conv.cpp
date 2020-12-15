#include "thorin/pass/fp/eta_conv.h"

#include "thorin/analyses/scope.h"

namespace thorin {

const Def* EtaConv::rewrite(const Def* def) {
    if (def->isa<Var>() || def->isa<Proxy>()) return def;

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nominal<Lam>(); !ignore(lam)) {
            if (!is_callee(def, i)) {
                if (expand_.contains(lam)) {
                    auto [j, ins] = def2exp_.emplace(def, nullptr);
                    if (ins) {
                        auto wrap = lam->stub(world(), lam->type(), lam->dbg());
                        wrappers_.emplace(wrap);
                        wrap->set_name(std::string("eta_wrap_") + lam->debug().name);
                        wrap->app(lam, wrap->var());
                        j->second = def->refine(i, wrap);
                        world().DLOG("eta-expansion '{}' -> '{}' using '{}'", def, j->second, wrap);
                    }

                    return j->second;
                }
            }

            if (wrappers_.contains(lam)) continue;

            if (auto app = lam->body()->isa<App>()) {
                if (app->arg() == lam->var() && !is_free(lam->var(), app->callee())) {
                    auto new_def = def->refine(i, app->callee());
                    world().DLOG("eta-reduction '{}' -> '{}' by eliminating '{}'", def, new_def, lam);
                    return new_def;
                }
            }
        }
    }

    return def;
}

undo_t EtaConv::analyze(const Def* def) {
    auto cur_lam = descend<Lam>(def);
    if (cur_lam == nullptr) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        undo = std::min(undo, analyze(def->op(i)));

        if (auto lam = def->op(i)->isa_nominal<Lam>(); !ignore(lam)) {
            if (is_callee(def, i)) {
                if (expand_.contains(lam)) continue;

                auto&& [l, u, ins] = insert<LamMap<Lattice>>(lam, Lattice::Callee);
                if (ins) {
                    world().DLOG("Bot -> Callee: '{}'", lam);
                    l = Lattice::Callee;
                } else if (l == Lattice::Once_Non_Callee) {
                    world().DLOG("Once_Non_Callee -> expand: '{}'", lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, u);
                }
            } else { // non-callee
                if (expand_.contains(lam)) {
                    undo = std::min(undo, cur_undo());
                    continue;
                }

                auto&& [l, u, ins] = insert<LamMap<Lattice>>(lam, Lattice::Once_Non_Callee);
                if (ins) {
                    world().DLOG("Bot -> Once_Non_Callee: '{}'", lam);
                } else {
                    if (l == Lattice::Callee)
                        world().DLOG("Callee -> expand: '{}'", lam);
                    else // l == Lattice::Once_Non_Callee
                        world().DLOG("Once_Non_Callee -> expand: '{}'", lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, u);
                }
            }
        }
    }

    return undo;
}

}
