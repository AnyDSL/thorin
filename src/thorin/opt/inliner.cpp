#include "thorin/opt/inliner.h"

#include "thorin/transform/mangle.h"

namespace thorin {

const Def* Inliner::rewrite(const Def* def) {
    if (auto param = def->isa<Param>())
        param->lam()->dump_head();

    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !lam->is_empty() && uses(lam) == 0) {
            ++uses(lam);
            return optimizer().rewrite(drop(app)->body());
        }
    }

    return def;
}

void Inliner::analyze(const Def* def) {
    if (def->isa<Param>()) return;
    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>(); lam && !lam->is_empty()) {
            ++uses(lam);
            if (uses(lam) > 1) {
                std::cout << "rollback" << std::endl;
                lam->dump_head();
            }
        }
    }
}

}
