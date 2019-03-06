#include "thorin/pass/inliner.h"

#include "thorin/transform/mangle.h"

namespace thorin {

// TODO here is another catch:
// Say you have sth like this
//  app(f, app(g, ...))
// Now, this code will inline g and set g's lattice to Dont_Inline.
// However, the inlined code might be dead after inlining f.

bool is_candidate(Lam* lam) { return lam != nullptr && !lam->is_empty() && !lam->is_external(); }

const Def* Inliner::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam)) {
            if (auto& info = lam2info(lam); info.lattice == Lattice::Bottom) {
                info.lattice = Lattice::Inlined_Once;
                info.undo = man().cur_state_id();
                man().new_state();
                return drop(app)->body();
            }
        }
    }

    return def;
}

void Inliner::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>(); is_candidate(lam)) {
            auto& info = lam2info(lam);
            if (info.lattice == Lattice::Bottom) {
                assertf(!def->isa<App>() || def->as<App>()->callee() == op, "this case should have been inlined");
                info.lattice = Lattice::Dont_Inline;
            } else if (info.lattice == Lattice::Inlined_Once) {
                info.lattice = Lattice::Dont_Inline;
                std::cout << "rollback: " << lam << std::endl;
                man().undo(info.undo);
            }
        }
    }
}

}
