#include "thorin/pass/partial_eval.h"

#include "thorin/rewrite.h"
#include "thorin/util/log.h"

namespace thorin {

const Def* PartialEval::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && lam->is_set()) {
            if (auto filter = isa_lit<bool>(thorin::rewrite(lam->filter(), lam->param(), app->arg())); filter && *filter) {
                outf("PE: {}\n", lam);
                return man().rewrite(thorin::rewrite(lam, app->arg()));
            }
        }
    }

    return def;
}

}
