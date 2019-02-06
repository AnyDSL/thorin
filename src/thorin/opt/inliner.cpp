#include "thorin/opt/inliner.h"

#include "thorin/transform/mangle.h"

namespace thorin {

const Def* Inliner::rewrite(const Def* def) {
    if (def->isa<Param>()) return def;

    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !lam->is_empty() && uses(lam) == 0)
            def = optimizer().rewrite(drop(app)->body());
    }

    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>())
            ++uses(lam);
    }

    return def;
}

}
