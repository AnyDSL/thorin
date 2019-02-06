#include "thorin/opt/inliner.h"

#include "thorin/transform/mangle.h"

namespace thorin {

void Inliner::enter(Lam*) {
}

const Def* Inliner::visit(const Def* def) {
    def->dump();
    for (auto op : def->ops()) {
        op = optimizer().lookup(op);
        if (auto lam = op->isa_lam())
            ++uses(lam);
    }

    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_lam(); lam && !lam->is_empty() && uses(lam) == 1) {
            auto new_lam = drop(app);
            return new_lam->body();
        }
    }

    return def;
}

}
