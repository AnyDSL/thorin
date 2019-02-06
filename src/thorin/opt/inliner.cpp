#include "thorin/opt/inliner.h"

#include "thorin/transform/mangle.h"

namespace thorin {

Def* Inliner::visit(Def*) {
    return nullptr;
}

const Def* Inliner::visit(const Def* def) {
    return def;
    if (def->isa<Param>()) return def;

    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>())
            ++uses(lam);
    }

    if (auto app = def->isa<App>()) {
        app->dump();
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !lam->is_empty() && uses(lam) == 1) {
            auto new_lam = drop(app);
            return optimizer().rewrite(new_lam->body());
        }
    }

    return def;
}

}
