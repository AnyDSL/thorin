#include "thorin/pass/fp/copy_prop.h"

#include "thorin/pass/fp/beta_red.h"
#include "thorin/pass/fp/eta_exp.h"

namespace thorin {

const Def* CopyProp::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto var_lam = app->callee()->isa_nom<Lam>(); !ignore(var_lam))
            return var2prop(app, var_lam);
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
        } else if (app->arg(i)->contains_proxy()) {
            world().DLOG("found proxy within app: {}@{}", var_lam, app);
            return app; // wait till proxy is gone
        } else if (args[i] == nullptr) {
            args[i] = app->arg(i);
        } else if (args[i] != app->arg(i)) {
            proxy_ops.emplace_back(var_lam->var(i));
        }
    }

    world().DLOG("app->args(): {, }", app->args());
    world().DLOG("args: {, }", args);
    world().DLOG("new_args: {, }", new_args);

    if (proxy_ops.size() > 1) {
        auto p = proxy(app->type(), proxy_ops, 0);
        world().DLOG("copxy: '{}': {, }", p, proxy_ops);
        return p;
    }

    assert(new_args.size() < var_lam->num_vars());
    auto&& [prop_lam, old_args] = var2prop_[var_lam];
    if (prop_lam == nullptr || old_args != args) {
        old_args = args;
        auto prop_dom = world().sigma(types);
        auto new_type = world().pi(prop_dom, var_lam->codom());
        prop_lam = var_lam->stub(world(), new_type, var_lam->dbg());
        beta_red_->keep(prop_lam);
        eta_exp_->new2old(prop_lam, var_lam);
        keep_.emplace(prop_lam); // don't try to propagate again
        world().DLOG("var_lam => prop_lam: {}: {} => {}: {}", var_lam, var_lam->type()->dom(), prop_lam, prop_dom);

        size_t j = 0;
        Array<const Def*> new_vars(app->num_args(), [&, prop_lam = prop_lam](size_t i) {
            return keep_.contains(var_lam->var(i)) ? prop_lam->var(j++) : args[i];
        });
        prop_lam->set(var_lam->apply(world().tuple(new_vars)));
    } else {
        world().DLOG("reuse var_lam => prop_lam: {}: {} => {}: {}", var_lam, var_lam->type()->dom(), prop_lam, prop_lam->type()->dom());
    }

    return app->world().app(prop_lam, new_args, app->dbg());
}

undo_t CopyProp::analyze(const Proxy* proxy) {
    auto var_lam = proxy->op(0)->as_nom<Lam>();
    world().DLOG("found proxy: {}", var_lam);

    for (auto op : proxy->ops().skip_front()) {
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

}
