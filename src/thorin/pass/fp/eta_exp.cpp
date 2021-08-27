#include "thorin/pass/fp/eta_exp.h"
#include "thorin/pass/fp/eta_red.h"

namespace thorin {

const Def* EtaExp::rewrite(const Def* def) {
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
        }
    }

    // if a wrapper is somehow reinstantiated again in a different expression, redo eta-expansion
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
            if (auto subst = wrap2subst_.lookup(lam)) {
                auto [orig, subst_def] = *subst;
                if (def != subst_def) {
                    assert(lam->body()->isa<App>() && lam->body()->as<App>()->callee() == orig);
                    return reexpand(def);
                }
            }
        }
    }

    return def;
}

Lam* EtaExp::eta_wrap(Lam* lam) {
    auto wrap = lam->stub(world(), lam->type(), lam->dbg());
    wrap->set_name(std::string("eta_") + lam->debug().name);
    wrap->app(lam, wrap->var());
    if (eta_red_) eta_red_->mark_irreducible(wrap);
    return wrap;
}

// TODO cleanup + document eta-reexpand
const Def* EtaExp::reexpand(const Def* def) {
    std::vector<std::pair<Lam*, Lam*>> refinements;
    Array<const Def*> new_ops(def->num_ops());

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>()) {
            if (auto subst = wrap2subst_.lookup(lam)) {
                auto [orig, subst_def] = *subst;
                auto wrap = eta_wrap(orig);
                refinements.emplace_back(wrap, orig);
                new_ops[i] = wrap;
                continue;
            }
        }

        new_ops[i] = def->op(i);
    }

    auto new_def = def->rebuild(world(), def->type(), new_ops, def->dbg());

    for (auto [wrap, lam] : refinements)
        wrap2subst_[wrap] = std::pair(lam, new_def);

    return new_def;
}

undo_t EtaExp::analyze(const Def* def) {
    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
            if (expand_.contains(lam)) continue;

            if (isa_callee(def, i)) {
                auto [_, l] = *data().emplace(lam, Lattice::Callee).first;
                if (l == Lattice::Non_Callee_1) {
                    world().DLOG("Callee: Callee -> Expand: '{}'", lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, visit_undo(lam));
                } else {
                    world().DLOG("Callee: Bot/Callee -> Callee: '{}'", lam);
                }
            } else {
                auto [it, first] = data().emplace(lam, Lattice::Non_Callee_1);

                if (first) {
                    world().DLOG("Non_Callee: Bot -> Non_Callee_1: '{}'", lam);
                } else {
                    world().DLOG("Non_Callee: {} -> Expand: '{}'", lattice2str(it->second), lam);
                    expand_.emplace(lam);
                    undo = std::min(undo, visit_undo(lam));
                }
            }
        }
    }

    return undo;
}

}
