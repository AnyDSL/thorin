#include "thorin/pass/copy_prop.h"

#include "thorin/util.h"

namespace thorin {

void CopyProp::visit(Def* cur_nom, Def* nom) {
    auto [cur_lam, param_lam] = std::pair(cur_nom->isa<Lam>(), nom->isa<Lam>());
    if (!cur_nom || !param_lam || keep_.contains(param_lam)) return;

    param_lam = lam2param(param_lam);
    if (param_lam->is_intrinsic() || param_lam->is_external()) {
        keep_.emplace(param_lam);
        return;
    }

    // build a prop_lam where args has been propagated from param_lam
    auto&& [visit, _] = get<Visit>(param_lam);
    if (auto& prop_lam = visit.prop_lam; !prop_lam) {
        std::vector<const Def*> types;
        for (size_t i = 0, e = visit.args.size(); i != e; ++i) {
            if (auto arg = visit.args[i]; arg && arg->isa<Top>())
                types.emplace_back(arg->type());
        }

        auto prop_domain = merge_sigma(param_lam->domain(), types);
        prop_lam = world().lam(world().pi(prop_domain, param_lam->codomain()), param_lam->debug());
        man().mark_tainted(prop_lam);
        world().DLOG("param_lam => prop_lam: {}: {} => {}: {}", param_lam, param_lam->type()->domain(), prop_lam, prop_domain);
        preds_n_.emplace(prop_lam);
        prop2param_[prop_lam] = param_lam;
    }
}

void CopyProp::enter(Def* nom) {
    auto prop_lam = nom->isa<Lam>();
    if (!prop_lam) return;

    if (auto param_lam = prop2param(prop_lam)) {
        auto&& [visit, _] = get<Visit>(param_lam);
        Array<const Def*> new_params(visit.args.size(), [&](size_t i) {
            if (visit.args[i])
                return visit.args[i]->isa<Top>() ? param_lam->param(i) : visit.args[i];
            else
                return world().bot(param_lam->param(i)->type());
        });
        auto new_param = world().tuple(new_params);
        man().map(param_lam->param(), new_param);
        prop_lam->set(param_lam->ops());
    }
}

const Def* CopyProp::rewrite(Def*, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto param_lam = app->callee()->isa_nominal<Lam>()) {
            auto&& [visit, _] = get<Visit>(param_lam);
            if (auto& prop_lam = visit.prop_lam) {
                std::vector<const Def*> args;
                for (size_t i = 0, e = visit.args.size(); i != e; ++i) {
                    if (visit.args[i] && visit.args[i]->isa<Top>()) args.emplace_back(app->arg(i));
                }

                return world().app(prop_lam, args);
            }
        }
    }

    return def;
}

undo_t CopyProp::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (!cur_lam || def->isa<Param>()) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);
        if (auto lam = op->isa<Lam>()) {
            return lam->gid();
        }
    }

    return undo;
}

}
