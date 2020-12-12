#include "thorin/pass/rw/partial_eval.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* PartialEval::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && lam->is_set()) {
            if (lam->filter() == world().lit_false()) return def; // optimize this common case

            auto [filter, body] = lam->apply(app->arg()).to_array<2>();
            if (auto f = isa_lit<bool>(filter); f && *f) {
                world().DLOG("PE {} within {}", lam, cur_nom());
                return body;
            }
        }
    }

    return def;
}

}
