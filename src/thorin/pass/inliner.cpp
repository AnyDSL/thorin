#include "thorin/pass/inliner.h"

#include "thorin/rewrite.h"
#include "thorin/util/log.h"

namespace thorin {

// TODO here is another catch:
// Say you have sth like this
//  app(f, app(g, ...))
// Now, this code will inline g and set g's lattice to Dont_Inline.
// However, the inlined code might be dead after inlining f.

bool is_candidate(Lam* lam) { return lam != nullptr && !lam->is_empty() && !lam->is_external(); }

const Def* Inliner::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam)) {
            if (lam2info(lam).lattice == Lattice::Bottom) {
                man().new_state();
                lam2info(lam).lattice = Lattice::Inlined_Once;
                outf("inline: {}\n", lam);
                return man().rewrite(drop(lam, app->arg()));
            }
        }
    }

    return def;
}

size_t Inliner::analyze(const Def* def) {
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
                std::cout << "rollback: " << lam << std::endl;
                keep_.emplace(lam);
                return info.undo;
            }
        }
    }

    return No_Undo;
}

}
