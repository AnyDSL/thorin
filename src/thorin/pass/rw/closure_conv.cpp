
#include "closure_conv.h"

namespace thorin {

const Def* ClosureConv::rewrite_rec(const Def*  def) {
    auto type = rewrite_rec(def->type());
    auto dbg = rewrite_rec(def->dbg());
    if (auto nom = def->as_nom()) {
        return rewrite(nom, type, dbg);
    } else {
        auto ops = Array<const Def*>(def->num_ops(), [&](auto i) {
            return rewrite_rec(def->op(i));
        });
        return rewrite(def, type, ops, dbg);
    }
}

template<bool rewrite_args>
const Pi* ClosureConv::lifted_fn_type(const Pi* pi, const Def* env_type) {
    assert(!pi->is_returning());
    auto rewrite_type = [&](auto type) {
        if constexpr(rewrite_args) { return rewrite_rec(type); } else { return type; }
    };
    auto dom = Array<const Def*>(pi->doms().size() + 1, [&](auto i) { 
        return (i == 0) ? env_type : rewrite_type(pi->dom(i) - 1); 
    });
    return world().cn(dom);
}

template<bool rewrite_args>
Sigma* ClosureConv::closure_type(const Pi* pi) {
    auto sigma = world().nom_sigma(world().kind(), 1, world().dbg("cconv_package_type"));
    auto tuple = lifted_fn_type<rewrite_args>(pi, sigma->var());
    sigma->set(0, tuple);
    mark(sigma, DONE);
    return sigma;
}

const Def* ClosureConv::closure_stub(Lam* lam) {
    auto fvs = Scope(lam).free_defs();
    auto env_types = Array<const Def*>(fvs.size());
    auto env_vars = Array<const Def*>(fvs.size());
    auto i = 0; 
    for (auto v: fvs) {
        env_types[i] = rewrite_rec(v->type());
        env_vars[i] = rewrite_rec(v);
        i++;
    }
    auto env_type = world().sigma(env_types);
    auto env = world().tuple(env_vars);

    auto name = lam->name();
    auto lifted_lam_type = lifted_fn_type<true>(lam->type(), env_type);
    auto debug = world().dbg(lam->debug());
    debug->set_name(name + "_cconv_lifted");
    auto lifted_lam = world().nom_lam(lifted_lam_type, lam->cc(), debug);
    lifted_lam->set_body(lam->body());
    mark(lifted_lam, CL_STUB);

    auto fv_map = std::make_unique<FVMap>(lam->var());
    auto env_param = lifted_lam->var(0_u64);
    i = 0;
    for (auto v: fvs) {
        fv_map->emplace(v, world().extract(env_param, i));
        i++;
    }
    fv_maps_.emplace(lifted_lam, std::move(fv_map));

    auto clos_type = closure_type<true>(lam->type());
    auto closure = world().tuple(clos_type, {env, lifted_lam}, 
            world().dbg(name + "_cconv_cl"));

    return closure;
}

void ClosureConv::enter() {
    auto stat = status(cur_nom());
    if (auto lam = man().cur_nom<Lam>(); lam && stat != DONE) {
        const Def* closure;
        if (stat == CL_STUB) {
            cur_fv_map_ = std::move(fv_maps_[lam]);
            closure = lam;
        } else {
            // toplevel def, should not have any fv's
            assert(lam->is_external());
            closure = closure_stub(lam);
            man().map(lam, closure);
        }
        rewrite_cur_nom_ = true;
        mark(closure, DONE);
        return;
    }
    rewrite_cur_nom_ = false;
}

const Def* ClosureConv::rewrite(Def* nom, const Def* type, const Def* dbg) {
    if (!should_rewrite(nom))         
        return nom;
    if (auto lam = nom->isa<Lam>()) {
        if (lam->type()->is_returning() || status(lam) != UNPROC)             
            return lam;
        else
            return closure_stub(lam);
    } else {
        // TODO: rewrite dep sigma
        // TODO: rewrite dep pi
        // TODO: rewrite dep array
        // TODO: rewrite dep pack
        return nom;
    }
}

const Def* ClosureConv::rewrite(const Def* def, const Def* type, Defs ops, const Def* dbg) {
    if (!should_rewrite(def))        
        return def;
    if (auto env_def = cur_fv_map_->lookup(def)) {
        return *env_def;
    } else if (auto pi = def->isa<Pi>()) {
        // rewrite cn[X..] -> closure(X..)
        if (!pi->is_returning())             
            return closure_type<false>(world().cn(ops[0]));
    } else if (auto app = def->isa<App>()) {
        // rewrite app (where callee = packed closure)
        auto callee = ops[0];
        if (!callee->type()->isa<Sigma>())
            return def;
        auto new_callee = world().extract(callee, 0_u64, world().dbg("cconv_proj_fn"));
        auto env = world().extract(callee, 1_u64, world().dbg("ccvonv_proj_env"));
        return world().app(new_callee, Array<const Def*>(ops.size(), [=](auto i) {
            return (i == 0) ? env : ops[i];
        }));
    } else if (def == cur_fv_map_->old_param()) {
        return cur_nom()->var();
    } else if (auto proj = def->isa<Extract>(); 
            proj && proj->tuple() == cur_fv_map_->old_param()) {
        // Shift param by one to account for new env param
        return world().extract(cur_nom()->var(),
                world().op(Wrap::add, WMode::none, ops[1], world().lit_nat_1()));
    }
    return def;
}

} 
