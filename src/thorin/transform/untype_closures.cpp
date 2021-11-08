#include <functional>

#include "thorin/transform/untype_closures.h"

namespace thorin {

static bool isa_ct(const Def* def, std::function<bool (const Def*)> var_pred) {
    if (def->num_ops() != 2)
        return false;
    auto cn = def->op(1)->isa<Pi>();
    return cn
        && var_pred(def->op(0))
        && cn->is_cn()
        && cn->num_ops() > 1
        && var_pred(cn->dom(0));
}

Sigma* UntypeClosures::isa_pct(const Def* def) {
    if (auto sigma = def->isa_nom<Sigma>())
        return isa_ct(def, [&](auto def) { return sigma->var() == def; }) ? sigma : nullptr;
    return nullptr;
}

const Sigma* UntypeClosures::isa_uct(const Def* def) {
    if (auto sigma = def->isa<Sigma>())
        return isa_ct(sigma, [](auto def) { return def == env_type(def->world()); })
             ? def->as<Sigma>() : nullptr;
    return nullptr;
}


void UntypeClosures::run() {
    auto externals = std::vector(world().externals().begin(), world().externals().end());
    for (auto [_, n]: externals)
        rewrite(n);
    while (!worklist_.empty()) {
        auto [lvm, lcm, lam] = worklist_.front();
        worklist_.pop();
        lcm_ = lcm;
        lvm_ = lvm;
        world().DLOG("in {} (lvm={}, lcm={})", lam, lvm_, lcm_);
        if (lam->is_set()) {
            lam->set_body(rewrite(lam->body()));
            lam->set_filter(rewrite(lam->filter()));
        }
    }
}

static const Def* get_mem_var(Lam *lam) {
    for (size_t i = 0; i < lam->num_doms(); i++)
        if (thorin::isa<Tag::Mem>(lam->dom(i)))
            return lam->var(i);
    assert(false && "Function \\wo :mem paramter");
}

Lam *UntypeClosures::make_stub(Lam* lam, bool unbox_env) {
    assert(lam && "make_stub: not a lam");
    auto new_type = world().cn(Array<const Def*>(lam->num_doms(), [&](auto i) {
        auto new_dom = rewrite(lam->dom(i));
        return (i == 0 && !unbox_env) ? world().type_ptr(new_dom) : new_dom;
    }));
    auto new_lam = lam->stub(world(), new_type, world().dbg("uc" + lam->name()));
    world().DLOG("stub {} ~> {}", lam, new_lam);
    new_lam->set_name(lam->name());
    new_lam->set_body(lam->body());
    new_lam->set_filter(lam->filter());
    if (lam->is_external()) {
        lam->make_internal();
        new_lam->make_external();
    }
    auto mem_var = get_mem_var(lam);
    const Def* lcm = get_mem_var(new_lam);
    const Def* env = new_lam->var(0_u64);
    if (!unbox_env) {
        auto env_mem = world().op_load(lcm, env);
        lcm = world().extract(env_mem, 0_u64);
        env = world().extract(env_mem, 1_u64, world().dbg("env"));
    }
    auto new_args = world().tuple(Array<const Def*>(lam->num_doms(), [&](auto i) {
        return (i == 0) ? env
             : (lam->var(i) == mem_var) ? lcm
             : new_lam->var(i);
    }));
    map(lam->var(), new_args);
    worklist_.emplace(mem_var, lcm, new_lam);
    return map<Lam>(lam, new_lam);
}

static size_t repr_size(const Def* type, size_t inf) {
    if (auto size = thorin::isa_sized_type(type)) {
        return as_lit(size);
    } else if (auto sigma = type->isa<Sigma>()) {
        auto size = 0;
        for (size_t i = 0; i < sigma->num_ops(); i++)
            size += repr_size(sigma->op(0), inf);
        return size;
    } else if (auto arr = type->isa<Arr>(); arr && isa_lit(arr->shape())) {
        return as_lit(arr->shape()) * repr_size(arr->body(), inf);
    } else {
        return inf;
    }
}

bool UntypeClosures::unbox_env(const Def* type) {
    // return repr_size(type, 64 * 2) <= 64;
    return false;
}

const Def* UntypeClosures::rewrite(const Def* def) {
    switch(def->node()) {
        case Node::Bot:
        case Node::Top:
        case Node::Kind:
        case Node::Space:
        case Node::Nat:
            return def;
    }

    if (auto new_def = old2new_.lookup(def))
        return *new_def;

    world().DLOG("rewrite {}", def);

    auto new_type = rewrite(def->type());
    auto new_dbg = def->dbg() ? rewrite(def->dbg()) : nullptr;

    if (auto pct = isa_pct(def)) {
        map(pct->var(), env_type());
        return map(def, world().sigma({rewrite(pct->op(0)), rewrite(pct->op(1))}));
    } else if (def->isa<Tuple>() && isa_pct(def->type())) {
        auto env = rewrite(def->op(0_u64));
        auto unbox = unbox_env(env->type());
        auto fn = make_stub(def->op(1_u64)->isa_nom<Lam>(), unbox);
        if (!unbox) {
            auto mem_ptr = world().op_alloc(env->type(), lcm_);
            auto mem = world().extract(mem_ptr, 0_u64);
            auto env_ptr = world().extract(mem_ptr, 1_u64, world().dbg(fn->name() + "env_ptr"));
            lcm_ = world().op_store(mem, env_ptr, env);
            map(lvm_, lcm_);
            auto ucl = world().tuple({env_ptr, fn});
            return world().op_bitcast(new_type, ucl);
        } else {
            return world().tuple({env, fn});
        }
    } else if (auto lam = def->isa_nom<Lam>()) {
        // Lam's in callee pos are scalarized (unpacked env)
        // or external in which case their env is []
        return make_stub(lam, true);
    } else if (auto nom = def->isa_nom()) {
        auto new_nom = nom->stub(world(), new_type, new_dbg);
        map(nom, new_nom);
        for (size_t i = 0; i < nom->num_ops(); i++)
            if (nom->op(i))
                new_nom->set(i, rewrite(nom->op(i)));
        return new_nom;
    } else {
        auto new_ops = Array<const Def*>(def->num_ops(), [&](auto i) {
            return rewrite(def->op(i));
        });
        
        auto new_def = def->rebuild(world(), new_type, new_ops, new_dbg);

        // We may need to update the mem token after all ops have been rewritten:
        // F (m, a1, ..., (env, f):pct)
        // ~>
        // let [m', env_ptr] = :alloc T m' 
        // let m'' = :store env_ptr env
        // F (m, a1, ..., (env_ptr, f'))
        // ~>
        // let ... 
        // F (m'', a1, ..., (env_ptr, f')
        for (size_t i = 0; i < new_def->num_ops(); i++)
            if (new_def->op(i)->type() == world().type_mem())
                new_def = new_def->refine(i, lcm_);

        if (new_type == world().type_mem()) { // :store
            lcm_ = new_def;
            lvm_ = def;
        } else if (new_type->isa<Sigma>()) {  // :alloc, :slot, ...
            for (size_t i = 0; i < new_type->num_ops(); i++) {
                if (new_type->op(i) == world().type_mem()) {
                    lcm_ = world().extract(new_def, i);
                    lvm_ = world().extract(def, i);
                }
            }
        }

        return map(def, new_def);
    }
}

}
