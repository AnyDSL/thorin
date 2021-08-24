#include "thorin/pass/fp/eta_red.h"

namespace thorin {

/// For now, only η-reduce <code>lam x.e x</code> if e is a @p Var or a @p Lam.
static const App* eta_rule(Lam* lam) {
    if (auto app = lam->body()->isa<App>()) {
        if (app->arg() == lam->var() && (app->callee()->isa<Var>() || app->callee()->isa<Lam>()))
            return app;
    }
    return nullptr;
}

const Def* EtaRed::rewrite(const Def* def) {
    if (def->isa<Var>() || def->isa<Proxy>()) return def;

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
            if (auto app = eta_rule(lam); app && !irreducible_.contains(lam)) {
                data().emplace(lam, std::pair(Lattice::Reduce, cur_undo()));
                auto new_def = def->refine(i, app->callee());
                world().DLOG("eta-reduction '{}' -> '{}' by eliminating '{}'", def, new_def, lam);
                return new_def;
            }
        }
    }

    return def;
}

undo_t EtaRed::analyze(const Def* def) {
    if (auto var = def->isa<Var>()) {
        if (auto lam = var->nom()->isa_nom<Lam>()) {
            auto [l, undo] = data().emplace(lam, std::pair(Lattice::Bot, cur_undo())).first->second;
            auto succ = irreducible_.emplace(lam).second;

            if (l == Lattice::Reduce && succ) {
                world().DLOG("irreducible: {}; found {}", lam, var);
                return undo;
            }
        }
    }

    return No_Undo;
}

}
