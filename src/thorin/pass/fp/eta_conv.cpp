#include "thorin/pass/fp/eta_conv.h"

#include "thorin/analyses/scope.h"

namespace thorin {

static bool is_free(const Param* param, const Def* def) {
    auto lam = param->nominal()->as<Lam>();

    // optimize common cases
    if (def == param) return true;
    for (auto p : lam->params())
        if (p == param) return true;

    Scope scope(lam);
    return scope.contains(def);
}

const Def* EtaConv::rewrite(Def*, const Def* def) {
    if (def->isa<Param>() || def->isa<Proxy>()) return def;

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nominal<Lam>(); !ignore(lam)) {
            if (!is_callee(def, i)) {
                if (expand_.contains(lam)) {
                    auto [j, ins] = def2exp_.emplace(def, nullptr);
                    if (ins) {
                        auto wrap = lam->stub(world(), lam->type(), lam->debug());
                        wrappers_.emplace(wrap);
                        wrap->set_name(std::string("eta_wrap_") + lam->name());
                        wrap->app(lam, wrap->param());
                        j->second = def->refine(i, wrap);
                        world().DLOG("eta-expansion '{}' -> '{}' using '{}'", def, j->second, wrap);
                    }

                    return j->second;
                }
            }

            if (wrappers_.contains(lam)) continue;

            if (auto app = lam->body()->isa<App>()) {
                if (app->arg() == lam->param() && !is_free(lam->param(), app->callee())) {
                    auto new_def = def->refine(i, app->callee());
                    world().DLOG("eta-reduction '{}' -> '{}' by eliminating '{}'", def, new_def, lam);
                    return new_def;
                }
            }
        }
    }

    return def;
}

undo_t EtaConv::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = descend(cur_nom, def);
    if (cur_lam == nullptr) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        undo = std::min(undo, analyze(cur_nom, def->op(i)));

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
