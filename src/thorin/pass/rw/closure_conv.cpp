#include "thorin/pass/rw/closure_conv.h"

#include "thorin/rewrite.h"
#include "thorin/analyses/scope.h"

namespace thorin {

const Def* ClosureConv::rewrite_rec(const Def*  def) {
    if (auto new_def = man().lookup(def)) {
        if (def == *new_def)
            return def;
        else
            def = *new_def;
    }
    switch (def->node()) {
        case Node::Kind: 
        case Node::Space:
        case Node::Nat:
        case Node::Top:
        case Node::Bot:
        case Node::Lit:
            return def;
        default: {
            auto type = rewrite_rec(def->type());
            auto dbg = (def->dbg()) ? rewrite_rec(def->dbg()) : nullptr;
            const Def *new_def;
            if (auto nom = def->isa_nom()) {
                new_def = rewrite(nom, type, dbg);
            } else {
                auto ops = Array<const Def*>(def->num_ops(), [&](auto i) {
                    return rewrite_rec(def->op(i));
                });
                new_def = rewrite(def, type, ops, dbg);
                if (new_def == def) {
                    new_def = def->rebuild(world(), type, ops, dbg);
                    world().DLOG("after rebuild (ops = {}) --> {} : {}", ops, new_def, new_def->type());
                }
            }
            man().map(def, new_def);
            return new_def;
        }
    }
}

template<bool rewrite_args>
const Pi* ClosureConv::lifted_fn_type(const Pi* pi, const Def* env_type) {
    assert(pi->is_cn());
    auto dom = Array<const Def*>(pi->doms().size() + 1, [&](auto i) { 
        if (i == 0)
            return env_type;
        else if constexpr(rewrite_args)
            return rewrite_rec(pi->dom(i - 1));
        else
            return pi->dom(i - 1);
    });
    // create a nom cn[] to memorize it
    auto new_pi = world().nom_pi(world().kind(), dom, world().dbg("ct"));
    new_pi->set_codom(world().bot_kind());
    mark(new_pi, DONE);
    return new_pi;
}

template<bool rewrite_args>
Sigma* ClosureConv::closure_type(const Pi* pi) {
    if (auto def = man().lookup(pi); def && (*def)->isa_nom<Sigma>())
        return (*def)->isa_nom<Sigma>();
    auto sigma = world().nom_sigma(world().kind(), 2, world().dbg("pct"));
    auto fn = lifted_fn_type<rewrite_args>(pi, sigma->var());
    sigma->set(0, sigma->var());
    sigma->set(1, fn);
    mark(sigma, DONE);
    return sigma;
}

const Def* ClosureConv::closure_stub(Lam* lam) {
    auto scope = Scope(lam);
    auto fvs = scope.free_defs();
    fvs.erase(lam);
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
    debug->set_name(name + "_closure");
    auto lifted_lam = world().nom_lam(lifted_lam_type, lam->cc(), debug);
    if (lam->is_external()) {
        lifted_lam->make_external();
    }

    push_old_param(lam, lifted_lam);
    auto env_param = lifted_lam->var(0_u64);
    auto rewriter = Rewriter(world(), &scope);
    i = 0;
    for (auto v: fvs) {
       rewriter.old2new.emplace(v, world().extract(env_param, i));
       i++;
    }
    lifted_lam->set_body(rewriter.rewrite(lam->body()));
    lifted_lam->set_filter(rewriter.rewrite(lam->filter()));
    mark(lifted_lam, CL_STUB);

    auto clos_type = closure_type<true>(lam->type());
    auto closure = world().tuple(clos_type, {env, lifted_lam}, 
            world().dbg(name + "pkd_closure"));

    world().DLOG("closure stub -- {} : {} --> lifted: {} : {} | closure {} : {}",
            lam, lam->type(), lifted_lam, lifted_lam_type, closure, clos_type);

    return closure;
}

void ClosureConv::enter() {
    auto stat = status(cur_nom());
    if (auto lam = man().cur_nom<Lam>(); lam && stat == CL_STUB) {
        cur_old_param_ = old_params_[lam];
        assert(cur_old_param_ != nullptr);
        rewrite_cur_nom_ = true;
        mark(lam, DONE);
    } else {
        rewrite_cur_nom_ = false;
    }
    world().DLOG("enter -- {}, status = {}, old_param: {}, rw? {}", cur_nom(), stat,
            cur_old_param_, rewrite_cur_nom_);
}

const Def* ClosureConv::rewrite(Def* nom, const Def* type, const Def* dbg) {
    world().DLOG("rewrite nom -- {} : {} -- {}", nom, type, status(nom));
    if (!should_rewrite(nom))         
        return nom;
    if (auto lam = nom->isa<Lam>()) {
        if (!lam->type()->is_cn() || status(lam) != UNPROC)             
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
    world().DLOG("cc rewrite struct: {} : {}\nops = {}", def, type, ops);
    if (!should_rewrite(def))        
        return def;
    if (auto pi = def->isa<Pi>()) {
        // rewrite cn[X..] -> closure(X..)
        if (pi->is_cn())             
            return closure_type<false>(world().cn(ops[0]));
    } else if (def->isa<App>()) {
        // rewrite app (where callee = packed closure)
        auto callee = ops[0];
        auto arg = ops[1];
        if (!callee->type()->isa<Sigma>())
            return def;
        auto env = world().extract(callee, 0_u64, world().dbg("closure_app_env"));
        auto new_callee = world().extract(callee, 1_u64, world().dbg("closure_app_fn"));
        world().DLOG("cc: unpack callee {} ~> fct = {} : {} | env ~> {} : {}", callee, new_callee, new_callee->type(), 
                env, env->type());
        auto type = new_callee->type()->isa<Pi>();
        assert(type);
        assert (type->dom(0) == env->type());
        auto new_arg = world().tuple(Array<const Def*>(type->num_doms(), [=](auto i) {
            return (i == 0) ? env : world().extract(arg, i - 1);
        }));
        return world().app(new_callee, new_arg);
    } else if (def->isa<Var>()) {
        if (def== old_param())
            return cur_nom()->var();
    } else if (auto proj = def->isa<Extract>(); proj && proj->tuple() == old_param()) {
        // Shift param by one to account for new env param
        auto idx = ops[1]->isa<Lit>();
        assert(idx);
        return world().extract(cur_nom()->var(), idx->fields() + 1);
    }
    return def;
}
} 
