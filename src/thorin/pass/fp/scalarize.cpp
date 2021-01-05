#include "thorin/pass/fp/scalarize.h"

namespace thorin {

const Def* Scalerize::rewrite(const Def* def) {
    auto app = def->isa<App>();
    if (app == nullptr) return def;

    auto tup_lam = app->callee()->isa_nom<Lam>();
    if (ignore(tup_lam) || tup_lam->num_vars() <= 1 || keep_.contains(tup_lam)) return app;

    auto& sca_lam = tup2sca_.emplace(tup_lam, nullptr).first->second;

    if (sca_lam == nullptr) {
        //auto&& [args, _, __] = insert<LamMap<Args>>(tup_lam);
        //args.resize(app->num_args());
        std::vector<const Def*> new_args;
        std::vector<const Def*> new_doms;

        for (size_t i = 0, e = app->num_args(); i != e; ++i) {
            auto a = tup_lam->num_outs();
            if (a == 0) continue; // remove empty tuples
            if (a == 1) continue; // keep
            if (keep_.contains(tup_lam->var(i))) {
                new_doms.emplace_back(tup_lam->var(i)->type());
                new_args .emplace_back(app->arg(i));
            } else {
                for (size_t j = 0; j != a; ++j) {
                    new_args.emplace_back(proj(tup_lam->var(i), a, j));
                    new_doms.emplace_back(proj(tup_lam->dom  (i), a, j));
                }
            }
        }
    }

    return def;
}

undo_t Scalerize::analyze(const Def* def) {
    auto cur_lam = descend<Lam>(def);
    if (cur_lam == nullptr) return No_Undo;

    return No_Undo;
    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nom<Lam>();
        if (keep_.emplace(lam).second) {
            world().DLOG("found proxy app of '{}'", lam);
            auto [undo, _] = put(lam);
            return undo;
        }
    } else {
        auto undo = No_Undo;
        for (auto op : def->ops()) {
            undo = std::min(undo, analyze(op));

            if (auto lam = op->isa_nom<Lam>(); !ignore(lam) && keep_.emplace(lam).second) {
            }
        }

        return undo;
    }

    return No_Undo;
}

}
