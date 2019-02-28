#include "thorin/pass/partial_eval.h"

#include "thorin/transform/mangle.h"

namespace thorin {

const Def* PartialEval::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !lam->is_empty()) {
            auto filter = isa_lit<bool>(lam->filter());
            if (*filter)
                return mgr().rebuild(drop(app)->body());
        }
    }

    return def;
}

}
