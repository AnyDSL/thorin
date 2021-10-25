#include "thorin/pass/fp/copy_prop.h"

namespace thorin {

const Def* CopyProp::rewrite(const Def* def) {
    auto app = def->isa<App>();
    if (app == nullptr) return def;

    auto var_lam = app->callee()->isa_nom<Lam>();
    if (ignore(var_lam) || var_lam->num_vars() == 0 || keep_.contains(var_lam)) return app;

    auto& args = data(var_lam);
    args.resize(app->num_args());
    std::vector<const Def*> new_args;
    std::vector<const Def*> types;

    bool update = false;
    bool changed = false;
    for (size_t i = 0, e = app->num_args(); i != e; ++i) {
        if (keep_.contains(var_lam->var(i))) {
            types.emplace_back(var_lam->var(i)->type());
            new_args.emplace_back(app->arg(i));
        } else if (args[i] == nullptr) {
            args[i] = app->arg(i);
            changed = true;
        } else if (args[i] != app->arg(i)) {
            keep_.emplace(var_lam->var(i));
            update = true;
        }
    }

    if (update) {
        if (new_args.size() == app->num_args()) keep_.emplace(var_lam);
        auto p = proxy(app->type(), app->ops(), 0);
        world().DLOG("proxy: '{}'", p);
        return p;
    }

    if (!changed) return def;

    auto& prop_lam = var2prop_[var_lam];
    if (prop_lam == nullptr || prop_lam->num_vars() != types.size()) {
        auto prop_dom = world().sigma(types);
        auto new_type = world().pi(prop_dom, var_lam->codom());
        prop_lam = var_lam->stub(world(), new_type, var_lam->dbg());
        keep_.emplace(prop_lam); // don't try to propagate again
        world().DLOG("var_lam => prop_lam: {}: {} => {}: {}", var_lam, var_lam->type()->dom(), prop_lam, prop_dom);

        size_t j = 0;
        Array<const Def*> new_vars(app->num_args(), [&](size_t i) {
            return keep_.contains(var_lam->var(i)) ? prop_lam->var(j++) : args[i];
        });
        prop_lam->set(var_lam->apply(world().tuple(new_vars)));
    }

    return app->world().app(prop_lam, new_args, app->dbg());
}

undo_t CopyProp::analyze(const Proxy* proxy) {
    auto lam = proxy->op(0)->as_nom<Lam>();
    world().DLOG("found proxy : {}", lam);
    return undo_visit(lam);
}

undo_t CopyProp::analyze(const Def* def) {
    return No_Undo;
    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>(); lam != nullptr && !ignore(lam) && keep_.emplace(lam).second) {
            //auto&& [_, u,ins] = data(lam);
            //if (!ins) {
                undo = std::min(undo, undo_visit(lam));
                world().DLOG("keep: {}", lam);
            //}
        }
    }

    return undo;
}

}
