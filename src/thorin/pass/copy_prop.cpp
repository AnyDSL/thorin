#include "thorin/pass/copy_prop.h"

#include "thorin/util.h"

namespace thorin {

const Def* CopyProp::rewrite(Def*, const Def* def) {
    auto app = def->isa<App>();
    if (app == nullptr) return def;

    auto param_lam = app->callee()->isa_nominal<Lam>();
    if (       param_lam == nullptr
            || param_lam->num_params() == 0
            || param_lam->is_external()
            || !param_lam->is_set()
            || prop2param_.contains(param_lam)
            || keep_.contains(param_lam))
        return app;

    auto&& [visit, _] = get(param_lam);
    visit.args.resize(app->num_args());
    std::vector<const Def*> new_args;
    std::vector<const Def*> types;

    bool update = false;
    for (size_t i = 0, e = app->num_args(); i != e; ++i) {
        if (keep_.contains(param_lam->param(i))) {
            types.emplace_back(param_lam->param(i)->type());
            new_args.emplace_back(app->arg(i));
        } else if (visit.args[i] == nullptr) {
            visit.args[i] = app->arg(i);
        } else if (visit.args[i] != app->arg(i)) {
            keep_.emplace(param_lam->param(i));
            update = true;
        }
    }

    if (update) {
        if (new_args.size() == app->num_args()) keep_.emplace(param_lam);
        return proxy(app->type(), app->ops());
    }

    auto& prop_lam = visit.prop_lam;
    if (prop_lam == nullptr) {
        auto prop_domain = world().sigma(types);
        auto new_type = world().pi(prop_domain, param_lam->codomain());
        prop_lam = param_lam->stub(world(), new_type, param_lam->debug());
        man().mark_tainted(prop_lam);
        prop2param_[prop_lam] = param_lam;
        world().DLOG("param_lam => prop_lam: {}: {} => {}: {}", param_lam, param_lam->type()->domain(), prop_lam, prop_domain);

        size_t j = 0;
        Array<const Def*> new_params(app->num_args(), [&](size_t i) {
            return keep_.contains(param_lam->param(i)) ? prop_lam->param(j++) : visit.args[i];
        });
        auto new_param = world().tuple(new_params);
        prop_lam->set(0_s, world().subst(param_lam->op(0), param_lam->param(), new_param));
        prop_lam->set(1_s, world().subst(param_lam->op(1), param_lam->param(), new_param));
    }

    return app->world().app(prop_lam, new_args, app->debug());
}

undo_t CopyProp::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (!cur_lam || def->isa<Param>()) return No_Undo;

    if (auto proxy = def->isa<Proxy>()) {
        if (proxy->index() == index())  {
            auto&& [_, undo] = get(proxy->op(0)->as_nominal<Lam>());
            return undo;
        }
        return No_Undo;
    }

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);
        if (auto lam = op->isa_nominal<Lam>()) {
            // if lam does not occur as callee - we can't do anything
            if ((!def->isa<App>() || i != 0)) {
                if (keep_.emplace(lam).second) {
                    auto&& [_, u] = get(lam);
                    world().DLOG("keep: {}", lam);
                    undo = std::min(undo, u);
                }
            }
        }
    }

    return undo;
}

}
