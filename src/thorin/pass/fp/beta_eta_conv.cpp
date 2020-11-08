#include "thorin/pass/fp/beta_eta_conv.h"

#include "thorin/rewrite.h"

namespace thorin {

static bool is_free(DefSet& done, const Param* param, const Def* def) {
    if (/* over-approximation */ def->isa_nominal() || def == param) return true;
    if (def->isa<Param>()) return false;

    if (done.emplace(def).second) {
        for (auto op : def->ops()) {
            if (is_free(done, param, op)) return true;
        }
    }

    return false;
}

static bool is_free(const Param* param, const Def* def) {
    DefSet done;
    return is_free(done, param, def);
}

const Def* BetaEtaConv::rewrite(Def*, const Def* def) {
    if (def->isa<Param>() || def->isa<Proxy>()) return def;

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nominal<Lam>(); lam != nullptr && !ignore(lam) && !man().is_tainted(lam)) {
            if (auto app = lam->body()->isa<App>(); app != nullptr && app->arg() == lam->param() && !is_free(lam->param(), app->callee())) {
                auto new_def = def->refine(i, app->callee());
                world().DLOG("eta-reduction '{}' -> '{}' by eliminating '{}'", def, new_def, lam);
                return new_def;
            }

            if (auto app = def->isa<App>(); app != nullptr && i == 0) {
                if (auto lam = app->callee()->isa_nominal<Lam>(); !ignore(lam) && !man().is_tainted(lam) && !keep_.contains(lam)) {
                    if (auto [_, ins] = put<LamSet>(lam); ins) {
                        world().DLOG("beta-redunction '{}'", lam);
                        return lam->apply(app->arg());
                    } else {
                        return proxy(app->type(), {lam, app->arg()});
                    }
                }
            } else {
                if (wrappers_.contains(def)) return def;

                auto& eta = def2eta_.emplace(def, nullptr).first->second;
                if (eta == nullptr) {
                    auto wrap = lam->stub(world(), lam->type(), lam->debug());
                    wrap->set_name(std::string("eta_wrap_") + lam->name());
                    wrap->app(lam, wrap->param());
                    eta = def->refine(i, wrap);
                    world().DLOG("eta-wrap '{}' -> '{}' using '{}'", def, eta, wrap);
                    wrappers_.emplace(eta);
                }

                return eta;
            }
        }
    }

    return def;
}


undo_t BetaEtaConv::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = descend(cur_nom, def);
    if (cur_lam == nullptr) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nominal<Lam>();
        if (keep_.emplace(lam).second) {
            world().DLOG("found proxy app of '{}'", lam);
            auto [undo, _] = put<LamSet>(lam);
            return undo;
        }
    } else {
        auto undo = No_Undo;
        for (auto op : def->ops()) {
            undo = std::min(undo, analyze(cur_nom, op));

            if (auto lam = op->isa_nominal<Lam>(); !ignore(lam) && !man().is_tainted(lam) && keep_.emplace(lam).second) {
                auto [lam_undo, ins] = put<LamSet>(lam);
                if (!ins) {
                    world().DLOG("non-callee-position of '{}'; undo to {} beta-reduction of {} within {}", lam, lam_undo, lam, cur_nom);
                    undo = std::min(undo, lam_undo);
                }
            }
        }

        return undo;
    }

    return No_Undo;
}

}
