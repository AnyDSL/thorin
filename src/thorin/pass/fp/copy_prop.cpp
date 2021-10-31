#include "thorin/pass/fp/copy_prop.h"

#include "thorin/pass/fp/eta_exp.h"

namespace thorin {

const Def* CopyProp::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto var_lam = app->callee()->isa_nom<Lam>(); !ignore(var_lam))
            return var2prop(app, var_lam);
    } else {
        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
                if (var2prop_.contains(lam))
                   return def->refine(i, proxy(lam->type(), {lam}, Etaxy));
            }
        }
    }

    return def;
}

const Def* CopyProp::var2prop(const App* app, Lam* var_lam) {
    if (ignore(var_lam) || var_lam->num_vars() == 0 || keep_.contains(var_lam)) return app;

    auto& args = data(var_lam);
    args.resize(app->num_args());
    DefVec new_args;
    DefVec types;
    DefVec proxy_ops = {var_lam};

    for (size_t i = 0, e = app->num_args(); i != e; ++i) {
        if (isa<Tag::Mem>(var_lam->var(i)->type())) {
            keep_.emplace(var_lam->var(i));
            types.emplace_back(var_lam->var(i)->type());
            new_args.emplace_back(app->arg(i));
            if (var_lam->num_vars() == 1) {
                keep_.emplace(var_lam);
                return app;
            }
        } else if (keep_.contains(var_lam->var(i))) {
            types.emplace_back(var_lam->var(i)->type());
            new_args.emplace_back(app->arg(i));
        } else if (args[i] == nullptr) {
            args[i] = app->arg(i);
        } else if (args[i] != app->arg(i)) {
            proxy_ops.emplace_back(var_lam->var(i));
        }
    }

    world().DLOG("args: {, }", args);
    world().DLOG("new_args: {, }", new_args);

    if (proxy_ops.size() > 1) {
        auto p = proxy(app->type(), proxy_ops, Copxy);
        world().DLOG("copxy: '{}': {, }", p, proxy_ops);
        return p;
    }

    assert(new_args.size() < var_lam->num_vars());
    auto&& [prop_lam, old_args] = var2prop_[var_lam];
    if (prop_lam == nullptr || old_args != args) {
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
        old_args = args;
    } else {
        world().DLOG("reuse var_lam => prop_lam: {}: {} => {}: {}", var_lam, var_lam->type()->dom(), prop_lam, prop_lam->type()->dom());
    }

    return app->world().app(prop_lam, new_args, app->dbg());
}

undo_t CopyProp::analyze(const Proxy* proxy) {
    if (auto etaxy = isa_proxy(proxy, Etaxy)) {
        auto etaxy_lam = etaxy->op(0)->as_nom<Lam>();
        eta_exp_->mark_expand(etaxy_lam, "copy_prop");
        world().DLOG("found etaxy '{}'", etaxy_lam);
        return undo_visit(etaxy_lam);
    } else if (auto copxy = isa_proxy(proxy, Copxy)) {
        auto var_lam = copxy->op(0)->as_nom<Lam>();
        world().DLOG("found copxy: {}", var_lam);

        for (auto op : copxy->ops().skip_front()) {
            if (op) {
                if (keep_.emplace(op).second) world().DLOG("keep var: {}", op);
            }
        }

        auto vars = var_lam->vars();
        if (std::all_of(vars.begin(), vars.end(), [&](const Def* def) { return keep_.contains(def); })) {
            if (keep_.emplace(var_lam).second)
                world().DLOG("keep var_lam: {}", var_lam);
        }

        return undo_visit(var_lam);
    }

    return No_Undo;
}

undo_t CopyProp::analyze(const Def* def) {
    return No_Undo;
    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nom<Lam>()) {
            if (!isa_callee(def, i) && !keep_.contains(lam) && var2prop_.contains(lam)) {
                eta_exp_->mark_expand(lam, "copy_prop");
                undo = std::min(undo, undo_visit(lam));
                world().DLOG("eta-expand: {}", lam);
            }
        }
    }

    return undo;
}

}
