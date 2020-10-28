#include "thorin/pass/inliner.h"

#include "thorin/rewrite.h"

namespace thorin {


std::variant<const Def*, undo_t> Inliner::rewrite(Def* cur_nom, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam)) {
            if (auto [undo, ins] = put<LamSet>(lam); !ins) {
                keep_.emplace(lam);
                world().DLOG("xxx: undo to {} inlinining of {} within {}", undo, lam, cur_nom);
                return undo;
            }

            world().DLOG("inline {} within {}", lam, cur_nom);
            return thorin::rewrite(lam, app->arg(), 1);
        }
    }

    return def;
}

undo_t Inliner::analyze(Def* cur_nom, const Def* def) {
    if (def->is_const() || analyzed(def) || def->isa<Param>()) return No_Undo;

    auto undo = No_Undo;
    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>(); is_candidate(lam) && keep_.emplace(lam).second) {
            auto [lam_undo, ins] = put<LamSet>(lam);
            if (!ins) {
                world().DLOG("yyy: undo to {} inlinining of {} within {}", lam_undo, lam, cur_nom);
                undo = std::min(undo, lam_undo);
            }
        }

        undo = std::min(undo, analyze(cur_nom, def));
    }

    return undo;
}

}
