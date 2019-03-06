#include "thorin/pass/partial_eval.h"

#include "thorin/rewrite.h"

namespace thorin {

// TODO remove
bool is_all_true(const Def* def) {
    if (def->isa<Tuple>()) {
        return std::all_of(def->ops().begin(), def->ops().end(), is_all_true);
    } else if (auto pack = def->isa<Pack>()) {
        return is_all_true(pack->body());
    } else {
        return is_type_bool(def->type()) && def->isa<Lit>() && def->as<Lit>()->box().get_bool();
    }
}

const Def* PartialEval::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>(); lam && !lam->is_empty()) {
            //auto filter = isa_lit<bool>(lam->filter());
            //if (filter && *filter)
            if (is_all_true(lam->filter()))
                return drop(lam, app->arg());
        }
    }

    return def;
}

}
