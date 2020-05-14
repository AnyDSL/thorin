#include "thorin/pass/inliner.h"

#include "thorin/rewrite.h"

namespace thorin {

static bool is_candidate(Lam* lam) { return lam != nullptr && lam->is_set() && !lam->is_external(); }

const Def* Inliner::rewrite(Def* cur_nom, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam)) {
            if (auto& info = lam2info(lam); info.lattice == Lattice::Bottom) {
                info.lattice = Lattice::Inlined_Once;
                info.undo = man().cur_state_id();
                world().DLOG("inline: {}", lam);
                return man().rewrite(cur_nom, thorin::rewrite(lam, app->arg(), 1));
            }
        }
    }

    return def;
}

size_t Inliner::analyze(Def* cur_lam, const Def* def) {
    if (def->isa<Param>()) return No_Undo;

    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>()) {
            if (keep_.contains(lam)) return No_Undo;

            auto& info = lam2info(lam);
            if (info.lattice == Lattice::Bottom) {
                assertf(!def->isa<App>() || def->as<App>()->callee() == op, "this case should have been inlined");
                info.lattice = Lattice::Dont_Inline;
            } else if (info.lattice == Lattice::Inlined_Once) {
                info.lattice = Lattice::Dont_Inline;
                world().DLOG("rollback: {} - {}", cur_lam, lam);
                keep_.emplace(lam);
                return info.undo;
            }
        }
    }

    return No_Undo;
}

}
