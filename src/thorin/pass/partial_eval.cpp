#include "thorin/pass/partial_eval.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* PartialEval::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !lam->is_empty()) {
            if (auto filter = isa_lit<bool>(thorin::rewrite(lam->filter(), lam->param(), app->arg())); filter && *filter)
                return drop(lam, app->arg());
        }
    }

    return def;
}

}
