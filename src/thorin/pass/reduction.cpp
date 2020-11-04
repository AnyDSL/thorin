#include "thorin/pass/reduction.h"

#include "thorin/rewrite.h"

namespace thorin {

static bool is_free(DefSet& done, const Param* param, const Def* def) {
    if (/* over-approximation */ def->isa_nominal() || def == param) return true;
    if (def->isa<Param>()) return false;

    if (done.emplace(def).second) {
        for (auto op : def->ops()) {
            if (is_free(done, param, op)) return true;
        }
    }

    return false;
}

static bool is_free(const Param* param, const Def* def) {
    DefSet done;
    return is_free(done, param, def);
}

const Def* Reduction::rewrite(Def* cur_nom, const Def* def) {
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nominal<Lam>(); lam != nullptr && !ignore(lam)) {
            auto app = lam->body()->isa<App>();

            //  η-reduction: λx.e x -> e, whenever x does not appear free in e
            if (app != nullptr && app->arg() == lam->param() && !is_free(lam->param(), app->callee())) {
                auto new_def = def->refine(i, app->callee());
                world().DLOG("eta-reduction '{}' -> '{}'", def, new_def);
                return new_def;
            }

            //if (app == nullptr || i != 0) {
                //Array<const Def*> new_ops(def->ops());
                //new_ops[i] = app->callee();
            //}
        }
    }

    if (auto app = def->isa<App>()) {
        //  β-reduction
        if (auto lam = app->callee()->isa_nominal<Lam>(); is_candidate(lam) && !keep_.contains(lam)) {
            if (auto [_, ins] = put<LamSet>(lam); ins) {
                world().DLOG("beta-redunction '{}' within '{}'", lam, cur_nom);
                return lam->apply(app->arg());
            } else {
                return proxy(app->type(), {lam, app->arg()});
            }
        }
    }

    return def;
}

undo_t Reduction::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = descend(cur_nom, def);
    if (cur_lam == nullptr) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nominal<Lam>();
        if (keep_.emplace(lam).second) {
            world().DLOG("found proxy app of '{}' within '{}'", lam, cur_nom);
            auto [undo, _] = put<LamSet>(lam);
            return undo;
        }
    } else {
        auto undo = No_Undo;
        for (auto op : def->ops()) {
            undo = std::min(undo, analyze(cur_nom, op));

            if (auto lam = op->isa_nominal<Lam>(); is_candidate(lam) && keep_.emplace(lam).second) {
                auto [lam_undo, ins] = put<LamSet>(lam);
                if (!ins) {
                    world().DLOG("non-callee-position of '{}'; undo to {} beta-reduction of {} within {}", lam, lam_undo, lam, cur_nom);
                    undo = std::min(undo, lam_undo);
                }
            }
        }

        return undo;
    }

    return No_Undo;
}

}
