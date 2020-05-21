#include "thorin/pass/partial_eval.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* PartialEval::rewrite(Def* cur_nom, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && lam->is_set() && !man().is_tainted(lam)) {
            if (lam->filter() == world().lit_false()) return def; // optimize this common case

            Scope scope(lam);
            if (auto filter = isa_lit<bool>(thorin::rewrite(lam, app->arg(), 0, scope)); filter && *filter) {
                world().DLOG("PE {} within {}", lam, cur_nom);
                return thorin::rewrite(lam, app->arg(), 1, scope);
            }
        }
    }

    return def;
}

}
