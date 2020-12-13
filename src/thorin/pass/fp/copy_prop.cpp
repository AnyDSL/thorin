#include "thorin/pass/fp/copy_prop.h"

namespace thorin {

const Def* CopyProp::rewrite(const Def* def) {
    auto app = def->isa<App>();
    if (app == nullptr) return def;

    auto param_lam = app->callee()->isa_nominal<Lam>();
    if (ignore(param_lam) || param_lam->num_params() == 0 || keep_.contains(param_lam)) return app;

    auto&& [args, _, __] = insert<LamMap<Args>>(param_lam);
    args.resize(app->num_args());
    std::vector<const Def*> new_args;
    std::vector<const Def*> types;

    bool update = false;
    bool changed = false;
    for (size_t i = 0, e = app->num_args(); i != e; ++i) {
        if (keep_.contains(param_lam->param(i))) {
            types.emplace_back(param_lam->param(i)->type());
            new_args.emplace_back(app->arg(i));
        } else if (args[i] == nullptr) {
            args[i] = app->arg(i);
            changed = true;
        } else if (args[i] != app->arg(i)) {
            keep_.emplace(param_lam->param(i));
            update = true;
        }
    }

    if (update) {
        if (new_args.size() == app->num_args()) keep_.emplace(param_lam);
        auto p = proxy(app->type(), app->ops(), 0);
        world().DLOG("proxy: '{}'", p);
        return p;
    }

    if (!changed) return def;

    auto& prop_lam = param2prop_[param_lam];
    if (prop_lam == nullptr || prop_lam->num_params() != types.size()) {
        auto prop_domain = world().sigma(types);
        auto new_type = world().pi(prop_domain, param_lam->codomain());
        prop_lam = param_lam->stub(world(), new_type, param_lam->dbg());
        keep_.emplace(prop_lam); // don't try to propagate again
        world().DLOG("param_lam => prop_lam: {}: {} => {}: {}", param_lam, param_lam->type()->domain(), prop_lam, prop_domain);

        size_t j = 0;
        Array<const Def*> new_params(app->num_args(), [&](size_t i) {
            return keep_.contains(param_lam->param(i)) ? prop_lam->param(j++) : args[i];
        });
        prop_lam->set(param_lam->apply(world().tuple(new_params)));
    }

    return app->world().app(prop_lam, new_args, app->dbg());
}

undo_t CopyProp::analyze(const Def* def) {
    auto cur_lam = descend<Lam>(def);
    if (cur_lam == nullptr) return No_Undo;

    if (auto proxy = isa_proxy(def)) {
        auto lam = proxy->op(0)->as_nominal<Lam>();
        auto&& [_, undo, __] = insert<LamMap<Args>>(lam);
        world().DLOG("found proxy : {}", lam);
        return undo;
    }

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        undo = std::min(undo, analyze(def->op(i)));

        if (auto lam = def->op(i)->isa_nominal<Lam>(); lam != nullptr && !ignore(lam) && keep_.emplace(lam).second) {
            auto&& [_, u,ins] = insert<LamMap<Args>>(lam);
            if (!ins) {
                undo = std::min(undo, u);
                world().DLOG("keep: {}", lam);
            }
        }
    }

    return undo;
}

}
