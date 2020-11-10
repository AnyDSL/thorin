#include "thorin/pass/fp/beta_red.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* BetaRed::rewrite(Def*, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam)) {
            if (auto [_, ins] = put<LamSet>(lam); ins) {
                world().DLOG("beta-reduction {}", lam);
                return lam->apply(app->arg());
            } else {
                return proxy(app->type(), {lam, app->arg()});
            }
        }
    }

    return def;
}

undo_t BetaRed::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = descend(cur_nom, def);
    if (cur_lam == nullptr) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nominal<Lam>();
        if (keep_.emplace(lam).second) {
            world().DLOG("found proxy app of '{}' within '{}'", lam, cur_nom);
            auto [undo, _] = put<LamSet>(lam);
            return undo;
        }
    } else {
        auto undo = No_Undo;
        for (auto op : def->ops()) {
            undo = std::min(undo, analyze(cur_nom, op));

            if (auto lam = op->isa_nominal<Lam>(); is_candidate(lam) && keep_.emplace(lam).second) {
                auto [lam_undo, ins] = put<LamSet>(lam);
                if (!ins) {
                    world().DLOG("non-callee-position of '{}'; undo to {} inlining of {} within {}", lam, lam_undo, lam, cur_nom);
                    undo = std::min(undo, lam_undo);
                }
            }
        }

        return undo;
    }

    return No_Undo;
}

}
