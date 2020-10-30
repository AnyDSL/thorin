#include "thorin/pass/reducer.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* Reducer::rewrite(Def* cur_nom, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam)) {
            if (auto [_, ins] = put<LamSet>(lam); ins) {
                world().DLOG("inline {} within {}", lam, cur_nom);
                return thorin::rewrite(lam, app->arg(), 1);
            } else {
                return proxy(app->type(), {lam, app->arg()});
            }
        }
    }

    return def;
}

undo_t Reducer::analyze(Def* cur_nom, const Def* def) {
    if (def->is_const() || analyzed(def) || def->isa<Param>()) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nominal<Lam>();
        if (keep_.emplace(lam).second) {
            world().DLOG("found proxy app of '{}' within '{}'", lam, cur_nom);
            auto [undo, _] = put<LamSet>(lam);
            return undo;
        }
    }

    auto undo = No_Undo;
    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>(); is_candidate(lam) && keep_.emplace(lam).second) {
            auto [lam_undo, ins] = put<LamSet>(lam);
            if (!ins) {
                world().DLOG("non-callee-position of '{}'; undo to {} inlining of {} within {}", lam, lam_undo, lam, cur_nom);
                undo = std::min(undo, lam_undo);
            }
        }

        undo = std::min(undo, analyze(cur_nom, def));
    }

    return undo;
}

}
