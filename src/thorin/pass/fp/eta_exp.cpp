#include "thorin/pass/fp/eta_exp.h"
#include "thorin/pass/fp/eta_red.h"

namespace thorin {

const Def* EtaExp::rewrite(const Def* def) {
    if (def->isa<Var>() || def->isa<Proxy>()) return def;

    auto eta_wrap = [&](Lam* lam) {
        auto wrap = lam->stub(world(), lam->type(), lam->dbg());
        wrap->set_name(std::string("eta_") + lam->debug().name);
        wrap->app(lam, wrap->var());
        eta_red_.irreducible_.emplace(wrap);
        return wrap;
    };

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
            if (isa_callee(def, i)) continue;

            if (expand_.contains(lam)) {
                auto [j, ins] = def2exp_.emplace(def, nullptr);
                if (ins) {
                    auto wrap = eta_wrap(lam);
                    auto new_def = def->refine(i, wrap);
                    wrap2subst_[wrap] = std::pair(lam, new_def);
                    j->second = new_def;
                    world().DLOG("eta-expansion '{}' -> '{}' using '{}'", def, j->second, wrap);
                }
                return j->second;
            }

            // if a wrapper is somehow reinstantiated again in a different expression, redo eta-expansion
            if (auto subst = wrap2subst_.lookup(lam)) {
                auto [orig, subst_def] = *subst;
                if (def != subst_def) {
                    auto wrap = eta_wrap(orig);
                    auto new_def = def->refine(i, wrap);
                    wrap2subst_[wrap] = std::pair(orig, new_def);
                    world().DLOG("eta-reexpand '{}' -> '{}'; lam: '{}', orig: '{}', wrap: '{}", def, new_def, lam, orig, wrap);
                    return new_def;
                }
            }
        }
    }

    return def;
}

undo_t EtaExp::analyze(const Def* def) {
    auto cur_lam = descend<Lam>(def);
    if (cur_lam == nullptr) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
            if (expand_.contains(lam)) continue;

            if (isa_callee(def, i)) {
                auto [l, u] = data().emplace(lam, std::pair(Lattice::Callee, cur_undo())).first->second;
                if (l == Lattice::Non_Callee_1) {
                    world().DLOG("Callee: Callee -> Expand: '{}'", lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, u);
                } else {
                    world().DLOG("Callee: Bot/Callee -> Callee: '{}'", lam);
                }
            } else {
                auto [it, first] = data().emplace(lam, std::pair(Lattice::Non_Callee_1, cur_undo()));
                auto [l, u] = it->second;

                if (first) {
                    world().DLOG("Non_Callee: Bot -> Non_Callee_1: '{}'", lam);
                } else {
                    world().DLOG("Non_Callee: {} -> Expand: '{}'", lattice2str(l), lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, u);
                }
            }
        }
    }

    return undo;
}

}
