#include "thorin/pass/partial_eval.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* PartialEval::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && lam->is_set()) {
            Scope scope(lam);
            if (auto filter = isa_lit<bool>(thorin::rewrite(lam, app->arg(), 0, scope)); filter && *filter) {
                world().DLOG("PE: {}", lam);
                return man().rewrite(thorin::rewrite(lam, app->arg(), 1, scope));
            }
        }
    }

    return def;
}

}
