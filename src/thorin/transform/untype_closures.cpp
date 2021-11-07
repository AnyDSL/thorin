
#include "thorin/transform/untype_closures.h"

namespace thorin {

Sigma* UntypeClosures::isa_pct(const Def* def) {
    if (def->num_ops() != 2)
        return nullptr;
    auto sigma = def->isa_nom<Sigma>();
    auto cn = def->op(1)->isa<Pi>();
    return (sigma && cn
        && sigma->op(0) == sigma->var()
        && cn->is_cn()
        && cn->num_ops() > 1
        && cn->op(1) == sigma->var()) ? sigma : nullptr;
}


void UntypeClosures::run() {
    for (auto [_, n]: world().externals())
        rewrite(n);
    while (!worklist_.empty()) {
        auto [lcm, lam] = worklist_.front();
        lcm_ = lcm;
        if (lam->is_set()) {
            lam->set_body(rewrite(lam->body()));
            lam->set_filter(rewrite(lam->filter()));
        }
    }
}

Lam *UntypeClosures::make_stub(Lam* lam, bool callee_pos) {
    assert(lam && "make_stub: not a lam");
    auto new_type = world().cn(Array<const Def*>(lam->num_doms(), [&](auto i) {
        auto new_dom = rewrite(lam->dom(i));
        return (i == 0 && !callee_pos) ? world().type_ptr(new_dom) : new_dom;
    }));
    auto new_lam = lam->stub(world(), new_type, world().dbg("uc" + lam->name()));
    new_lam->set_name(lam->name());
    if (lam->is_external()) {
        new_lam->make_external();
        lam->make_internal();
    }
    new_lam->set_body(lam->body());
    new_lam->set_filter(lam->filter());
    const Def* lcm;
    const Def* env;
    if (callee_pos) {
        auto env_mem = world().op_load(new_lam->mem_var(), new_lam->op(0_u64));
        env = world().extract(env_mem, 0_u64);
        lcm = world().extract(env_mem, 1_u64);
    } else {
        env = new_lam->var(0_u64);
        lcm = new_lam->mem_var();
    }
    auto new_args = world().tuple(Array<const Def*>(lam->num_doms(), [&](auto i) {
        return (i == 0) ? env
             : (lam->var(i) == lam->mem_var()) ? lcm
             : new_lam->var(i);
    }));
    map(lam->var(), new_args);
    worklist_.emplace(lcm, new_lam);
    return map<Lam>(lam, new_lam);
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

    auto new_type = rewrite(def->type());
    auto new_dbg = def->dbg() ? rewrite(def->dbg()) : nullptr;

    if (auto pct = isa_pct(def)) {
        map(pct->var(), env_type());
        return map(def, world().sigma({rewrite(pct->op(0)), rewrite(pct->op(1))}));
    } else if (def->isa<Tuple>() && isa_pct(def->type())) {
        auto fn = make_stub(def->op(1_u64)->isa_nom<Lam>(), true);
        auto env = rewrite(def->op(0_u64));
        auto env_type = world().type_ptr(rewrite(def->type()->op(0_u64)));
        auto mem_ptr = world().op_alloc(env_type, lcm_);
        auto env_ptr = mem_ptr->op(0_u64);
        auto mem = mem_ptr->op(1_u64);
        lcm_ = world().op_store(mem, env_ptr, env);
        map(lvm_, lcm_);
        auto ucl = world().tuple({env_ptr, fn});
        return world().op_bitcast(new_type, ucl);
    } else if (auto lam = def->isa_nom<Lam>()) {
        return make_stub(lam, false);
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
        // let [m', env_ptr] = :alloc T* m' 
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
