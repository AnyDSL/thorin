#include "thorin/pass/inliner.h"

#include "thorin/transform/mangle.h"

namespace thorin {

// TODO here is another catch:
// Say you have sth like this
//  app(f, app(g, ...))
// Now, this code will inline g and set g's lattice to Dont_Inline.
// However, the inlined code might be dead after inlining f.

const Def* Inliner::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !lam->is_empty()) {
            if (auto& inf = info(lam); inf.lattice == Lattice::Bottom) {
                inf.lattice = Lattice::Inlined_Once;
                inf.undo = mgr().num_states();
                std::cout << "inline: " << lam << std::endl;
                return mgr().rewrite(drop(app)->body());
            }
        }
    }

    return def;
}

void Inliner::analyze(const Def* def) {
    if (def->isa<Param>()) return;
    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>(); lam && !lam->is_empty()) {
            switch (auto& inf = info(lam); inf.lattice) {
                case Lattice::Inlined_Once:
                    inf.lattice = Lattice::Dont_Inline;
                    std::cout << "rollback: " << lam << std::endl;
                    mgr().undo(inf.undo);
                    break;
                default:
                    assertf(inf.lattice != Lattice::Bottom || (!def->isa<App>() || def->as<App>()->callee() != op), "this case should have been inlined");
                    inf.lattice = Lattice::Dont_Inline;
                    break;
            }
        }
    }
}

}
