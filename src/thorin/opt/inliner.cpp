#include "thorin/opt/inliner.h"

#include "thorin/transform/mangle.h"

namespace thorin {

void Inliner::enter(Lam*) {
}

const Def* Inliner::visit(const Def* def) {
    if (def->isa<Param>()) return def;

    for (auto op : def->ops()) {
        if (auto lam = op->isa_lam())
            ++uses(lam);
    }

    if (auto app = def->isa<App>()) {
        app->dump();
        if (auto lam = app->callee()->isa_lam(); lam && !lam->is_empty() && uses(lam) == 1) {
            auto new_lam = drop(app);
            return optimizer().rewrite(new_lam->body());
        }
    }

    return def;
}

}
