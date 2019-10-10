#include "thorin/pass/inliner.h"

#include "thorin/rewrite.h"

namespace thorin {

// TODO here is another catch:
// Say you have sth like this
//  app(f, app(g, ...))
// Now, this code will inline g and set g's lattice to Dont_Inline.
// However, the inlined code might be dead after inlining f.

bool is_candidate(Lam* lam) { return lam != nullptr && lam->is_set() && !lam->is_external(); }

const Def* Inliner::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam)) {
            if (auto& info = lam2info(lam); info.lattice == Lattice::Bottom) {
                info.lattice = Lattice::Inlined_Once;
                info.undo = man().cur_state_id();
                man().new_state();
                world().DLOG("inline: {}", lam);
                return man().rewrite(thorin::rewrite(lam, app->arg()).back());
            }
        }
    }

    return def;
}

void Inliner::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>()) {
            if (keep_.contains(lam)) return;

            auto& info = lam2info(lam);
            if (info.lattice == Lattice::Bottom) {
                assertf(!def->isa<App>() || def->as<App>()->callee() == op, "this case should have been inlined");
                info.lattice = Lattice::Dont_Inline;
            } else if (info.lattice == Lattice::Inlined_Once) {
                info.lattice = Lattice::Dont_Inline;
                world().DLOG("rollback: {}", lam);
                keep_.emplace(lam);
                man().undo(info.undo);
            }
        }
    }
}

}
