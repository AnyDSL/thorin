
#include "thorin/analyses/scope.h"
#include "thorin/transform/closure_conv.h"

namespace thorin {

void ClosureConv::run() {
    for (auto ext_def: world().externals()) {
        worklist_.emplace(ext_def);
    }
    world().DLOG("===== CC (run): start =====");
    while (!worklist_.empty()) {
        auto def = worklist_.front();
        worklist_.pop();
        if (auto lam = def->isa_nom<Lam>(); lam && lam->type()->is_cn()) {
            world().DLOG("===== CC (run): rewrite function {} =====", lam);
            auto closure = make_closure(lam);
            auto new_fn = closure.fn;
            auto env = closure.env;
            auto subst = Def2Def();
            auto env_param = new_fn->var(0_u64);
            for (size_t i = 0; i < env->num_ops(); i++) {
                auto fv = env->op(i);
                subst.emplace(env->op(i), world().extract(env_param, i, world().dbg("cc_fv")));
            }
            auto params = 
                world().tuple(Array<const Def*>(lam->num_doms(), [&] (auto i) {
                    return new_fn->var(i + 1); 
                }), world().dbg("cc_param"));
            subst.emplace(lam->var(), params);
            auto body = rewrite(new_fn->body(), &subst);
            new_fn->set_body(body);
            new_fn->dump(9000);
        }
        else {
            world().DLOG("CC (run): rewrite def {}", def);
            rewrite(def);
        }
        world().DLOG("===== ClosureConv: done ======");
        world().debug_stream();
    }
}


const Def *ClosureConv::rewrite(const Def *def, Def2Def *subst) {
    auto map = [&](const Def *new_def) {
        if (subst)
            subst->emplace(def, new_def);
        return new_def;
    };

    if (def->no_dep()) {
        return map(def);
    } else if (subst && subst->contains(def)) {
        return *subst->lookup(def);
    } else if (auto pi = def->isa_nom<Pi>(); pi && pi->is_cn()) {
        /* Types: rewrite dom, codom \w susbt here */
        return map(closure_type(pi));
    } else if (auto lam = def->isa_nom<Lam>(); lam && lam->type()->is_cn()) {
        auto [fv_env, fn] = make_closure(lam);
        auto closure_type = rewrite(lam->type(), subst);
        auto env = rewrite(fv_env, subst);
        auto closure = world().tuple(closure_type, {fn, env});
        world().DLOG("CC (rw): build closure: {} ~~> {} = (fn {}, env {}) : {}", lam, closure,
                fn, env, closure_type);
        return map(closure);
    } else if (auto nom = def->isa_nom()) {
        assert(false && "TODO: rewrite: handle noms");
    } else {
        auto new_type = rewrite(def->type(), subst);
        auto new_dbg = (def->dbg()) ? rewrite(def->dbg(), subst) : nullptr;
        auto new_ops = Array<const Def*>(def->num_ops(), [&](auto i) {
            return rewrite(def->op(i), subst);
        });
        if (auto app = def->isa<App>(); app && new_ops[1]->type()->isa<Sigma>()) {
            auto closure = new_ops[0];
            auto args = new_ops[1];
            auto fn = world().extract(closure, 0_u64, world().dbg("cc_app_env"));
            auto env = world().extract(closure, 1_u64, world().dbg("cc_app_f"));
            world().DLOG("CC (rw): call closure {}: APP {} {} {}", closure, fn, env, args);
            return map(world().app(fn, Array<const Def*>(args->num_ops() + 1, [&](auto i) {
                return (i == 0) ? env : world().extract(args, i - 1);
            })));
        } else {
            return map(def->rebuild(world(), new_type, new_ops, new_dbg));
        }
    }
}

const Def *ClosureConv::closure_type(const Pi *pi, const Def *env_type) {
    if (!env_type) {
        auto sigma = world().nom_sigma(world().kind(), 2_u64, world().dbg("cc_pct"));
        auto new_pi = closure_type(pi, sigma->var());
        sigma->set(0, sigma->var());
        sigma->set(1, new_pi);
        closure_types_.emplace(pi, sigma);
        world().DLOG("CC (cl_type): make pct: {} ~~> {}", pi, sigma);
        return sigma;
    } else {
        auto dom = world().sigma(Array<const Def*>(pi->num_doms() + 1, [&](auto i)  {
            return (i == 0) ? env_type : rewrite(pi->dom(i - 1));
        }));
        auto new_pi = world().cn(dom, world().dbg("cc_ct"));
        world().DLOG("CC (cl_type: make ct: {}, env = {} ~~> {})", pi, env_type, new_pi);
        return new_pi;
    }
}

ClosureConv::Closure ClosureConv::make_closure(Lam *fn) {
    if (auto closure = closures_.lookup(fn))
        return *closure;

    auto scope = Scope(fn);
    auto fv_set = scope.free_defs();
    fv_set.erase(fn);
    auto fvs = Array<const Def*>(fv_set.size());
    auto fvs_types = Array<const Def*>(fv_set.size());
    auto i = 0;
    for (auto fv: fv_set) {
        fvs[i] = fv;
        fvs_types[i] = fv->type();
        i++;
    }
    auto env = world().tuple(fvs);
    auto env_type = world().sigma(fvs_types);

    /* Types: rewrite function type here \w fv s */
    auto new_fn_type = closure_type(fn->type(), env_type)->as<Pi>();
    auto new_lam = world().nom_lam(new_fn_type, world().dbg("cc_" + fn->name()));
    new_lam->set_body(fn->body());
    if (fn->is_external()) {
        new_lam->make_external();
    }

    world().DLOG("CC (make_closure): {} : {} ~~> {} : {}, env = {} : {}", fn, fn->type(), new_lam,
            new_fn_type, env, env_type);

    auto closure = Closure{env, new_lam};
    closures_.emplace(fn, closure);
    worklist_.emplace(new_lam);
    return closure;
}

}
