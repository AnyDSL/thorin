
#include "thorin/analyses/scope.h"
#include "thorin/transform/closure_conv.h"

namespace thorin {

void ClosureConv::run() {
    auto externals = std::vector(world().externals().begin(), world().externals().end());
    for (auto [_, ext_def]: externals) {
        rewrite(ext_def);
    }
    world().DLOG("===== CC (run): start =====");
    while (!worklist_.empty()) {
        auto def = worklist_.front();
        worklist_.pop();
        if (auto closure = closures_.lookup(def)) {
            auto new_fn = closure->fn;
            auto env = closure->env;
            auto num_fvs = closure->num_fvs;
            auto old_fn = closure->old_fn;

            world().DLOG("===== CC (run): closure body {} [old={}, env={}] =====", 
                    new_fn, old_fn, env);
            auto subst = Def2Def();
            auto env_param = new_fn->var(0_u64);
            if (num_fvs == 1) {
                subst.emplace(env, env_param);
            } else {
                for (size_t i = 0; i < num_fvs; i++) {
                    subst.emplace(env->op(i), world().extract(env_param, i, world().dbg("cc_fv")));
                }
            }

            auto params = 
                world().tuple(Array<const Def*>(old_fn->num_doms(), [&] (auto i) {
                    return new_fn->var(i + 1); 
                }), world().dbg("cc_param"));
            subst.emplace(old_fn->var(), params);

            auto filter = (new_fn->filter()) 
                ? rewrite(new_fn->filter(), &subst) 
                : world().lit_false(); // extern function?
            
            auto body = (new_fn->body())
                ? rewrite(new_fn->body(), &subst)
                : world().app(old_fn, params); // extern function

            new_fn->set_body(body);
            new_fn->set_filter(filter);
        }
        else {
            world().DLOG("CC (run): rewrite def {}", def);
            rewrite(def);
        }
        world().DLOG("===== (run) done rewrite");
    }
    world().DLOG("===== ClosureConv: done ======");
    world().debug_stream();
}


const Def* ClosureConv::rewrite(const Def* def, Def2Def* subst) {
    switch(def->node()) {
        case Node::Kind:
        case Node::Space:
        case Node::Nat:
        case Node::Bot:
        case Node::Top:
        case Node::Axiom:
            return def;
        default:
            break;
    }

    auto map = [&](const Def* new_def) {
        if (subst)
            subst->emplace(def, new_def);
        return new_def;
    };

    if (subst && subst->contains(def)) {
        return map(*subst->lookup(def));
    } else if (auto pi = def->isa<Pi>(); pi && pi->is_cn()) {
        /* Types: rewrite dom, codom \w susbt here */
        return map(closure_type(pi));
    } else if (auto lam = def->isa_nom<Lam>(); lam && lam->type()->is_cn()) {
        auto [old, num, fv_env, fn] = make_closure(lam);
        auto closure_type = rewrite(lam->type(), subst);
        auto env = rewrite(fv_env, subst);
        auto closure = world().tuple(closure_type, {env, fn});
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
        if (auto app = def->isa<App>(); app && new_ops[0]->type()->isa<Sigma>()) {
            auto closure = new_ops[0];
            auto args = new_ops[1];
            auto env = world().extract(closure, 0_u64, world().dbg("cc_app_env"));
            auto fn = world().extract(closure, 1_u64, world().dbg("cc_app_f"));
            world().DLOG("CC (rw): call closure {}: APP {} {} {}", closure, fn, env, args);
            return map(world().app(fn, Array<const Def*>(args->num_ops() + 1, [&](auto i) {
                return (i == 0) ? env : world().extract(args, i - 1);
            })));
        } else {
            return map(def->rebuild(world(), new_type, new_ops, new_dbg));
        }
    }
}

const Def* ClosureConv::closure_type(const Pi* pi, const Def* env_type) {
    if (!env_type) {
        if (auto pct = closure_types_.lookup(pi))
            return* pct;
        auto sigma = world().nom_sigma(world().kind(), 2_u64, world().dbg("cc_pct"));
        auto new_pi = closure_type(pi, sigma->var());
        sigma->set(0, sigma->var());
        sigma->set(1, new_pi);
        closure_types_.emplace(pi, sigma);
        world().DLOG("CC (cl_type): make pct: {} ~~> {}", pi, sigma);
        return sigma;
    } else {
        auto dom = world().sigma(Array<const Def*>(pi->num_doms() + 1, [&](auto i) {
            return (i == 0) ? env_type : rewrite(pi->dom(i - 1));
        }));
        auto new_pi = world().cn(dom, world().dbg("cc_ct"));
        world().DLOG("CC (cl_type: make ct: {}, env = {} ~~> {})", pi, env_type, new_pi);
        return new_pi;
    }
}


void compute_fvs(Lam* fn, DefSet& visited, DefSet& fvs) {
    if (visited.contains(fn))
        return;
    visited.insert(fn);
    auto scope = Scope(fn);
    for (auto fv: scope.free_defs()) {
        if (fv == fn || fv->is_external() || fv->isa<Axiom>())
            continue;
        else if (auto callee = fv->isa_nom<Lam>())
            compute_fvs(callee, visited, fvs);
        else
            fvs.insert(fv);
    }
}

void compute_fvs(Lam* lam, DefSet &fvs) {
    auto visited = DefSet();
    compute_fvs(lam, visited, fvs);
}


ClosureConv::Closure ClosureConv::make_closure(Lam* fn) {
    if (auto closure = closures_.lookup(fn))
        return* closure;

    auto fv_set = DefSet();
    compute_fvs(fn, fv_set);
    auto fvs = std::vector<const Def*>();
    auto fvs_types = std::vector<const Def*>();
    for (auto fv: fv_set) {
        fvs.emplace_back(fv);
        fvs_types.emplace_back(rewrite(fv->type()));
    }
    auto env = world().tuple(fvs);
    auto env_type = world().sigma(fvs_types);

    /* Types: rewrite function type here \w fv s */
    auto new_fn_type = closure_type(fn->type(), env_type)->as<Pi>();
    auto new_lam = world().nom_lam(new_fn_type, world().dbg("cc_" + fn->name()));
    new_lam->set_body(fn->body());
    new_lam->set_filter(fn->filter());
    if (fn->is_external()) { 
        new_lam->make_external();
        if (fn->body() && fn->filter()) // imported external
            fn->make_internal();
    }

    world().DLOG("CC (make_closure): {} : {} ~~> {} : {}, env = {} : {}", fn, fn->type(), new_lam,
            new_fn_type, env, env_type);

    auto closure = Closure{fn, fv_set.size(), env, new_lam};
    closures_.emplace(fn, closure);
    closures_.emplace(new_lam, closure);
    worklist_.emplace(new_lam);
    return closure;
}

}
