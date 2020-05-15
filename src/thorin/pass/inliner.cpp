#include "thorin/pass/inliner.h"

#include "thorin/rewrite.h"

namespace thorin {

static bool is_candidate(Lam* lam) { return lam != nullptr && lam->is_set() && !lam->is_external(); }

const Def* Inliner::rewrite(Def* cur_nom, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam) && first_inline(lam)) {
            world().DLOG("inline: {} within {}", lam, cur_nom);
            return man().rewrite(cur_nom, thorin::rewrite(lam, app->arg(), 1));
        }
    }

    return def;
}

undo_t Inliner::analyze(Def*, const Def* def) {
    if (!def->isa<Param>()) {
        for (auto op : def->ops()) {
            if (auto lam = op->isa_nominal<Lam>()) {
                if (keep_.emplace(lam).second) {
                    if (auto undo = inlined_once(lam)) return *undo;
                }
            }
        }
    }

    return No_Undo;
}

}
