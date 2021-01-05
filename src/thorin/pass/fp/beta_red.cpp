#include "thorin/pass/fp/beta_red.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* BetaRed::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nom<Lam>(); !ignore(lam) && !keep_.contains(lam)) {
            if (auto [_, ins] = put(lam); ins) {
                world().DLOG("beta-reduction {}", lam);
                return lam->apply(app->arg()).back();
            } else {
                return proxy(app->type(), {lam, app->arg()}, 0);
            }
        }
    }

    return def;
}

undo_t BetaRed::analyze(const Def* def) {
    auto cur_lam = descend<Lam>(def);
    if (cur_lam == nullptr) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nom<Lam>();
        if (keep_.emplace(lam).second) {
            world().DLOG("found proxy app of '{}' within '{}'", lam, cur_lam);
            auto [undo, _] = put(lam);
            return undo;
        }
    } else {
        auto undo = No_Undo;
        for (auto op : def->ops()) {
            undo = std::min(undo, analyze(op));

            if (auto lam = op->isa_nom<Lam>(); !ignore(lam) && keep_.emplace(lam).second) {
                auto [lam_undo, ins] = put(lam);
                if (!ins) {
                    world().DLOG("non-callee-position of '{}'; undo to {} inlining of {} within {}", lam, lam_undo, lam, cur_lam);
                    undo = std::min(undo, lam_undo);
                }
            }
        }

        return undo;
    }

    return No_Undo;
}

}
