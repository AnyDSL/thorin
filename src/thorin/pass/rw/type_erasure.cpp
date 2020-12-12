#include "thorin/pass/rw/type_erasure.h"

namespace thorin {

Def* TypeErasure::rewrite(Def*, Def*, const Def*, const Def*) {
    return nullptr;
}

const Def* TypeErasure::rewrite(Def*, const Def* old_def, const Def*, Defs new_ops, const Def* new_dbg) {
    if (auto vel = old_def->isa<Vel>()) {
        auto join  = vel->type()->as<Join>();
        auto value = new_ops[0];
        auto sigma = join->convert();
        auto val   = world().op_bitcast(sigma->op(1), value, new_dbg);
        return world().tuple(sigma, {world().lit_int(join->num_ops(), join->find(vel->value()->type())), val});
    } else if (auto test = old_def->isa<Test>()) {
        auto [value, probe, match, clash] = new_ops.to_array<4>();
        auto [index, box] = value->split<2>();

        auto join = test->value()->type()->as<Join>();
        auto mpi = match->type()->as<Pi>();
        auto dom = mpi->domain()->out(0);
        auto wpi = world().pi(dom, mpi->codomain());
        auto wrap = world().nom_lam(wpi, world().dbg("wrap_match"));
        auto probe_i = join->index(probe);
        wrap->app(match, {wrap->param(), world().op_bitcast(probe, box)});
        auto cmp = world().op(ICmp::e, index, probe_i);
        return world().select(wrap, clash, cmp, new_dbg);
    } else if (auto et = old_def->isa<Et>()) {
        return world().tuple(et->type()->as<Meet>()->convert(), new_ops, new_dbg);
    } else if (auto pick = old_def->isa<Pick>()) {
        auto meet = pick->value()->type()->as<Meet>();
        auto index = meet->index(pick->type());
        return world().extract(new_ops[0], index, new_dbg);
    }

    return nullptr;
}

const Def* TypeErasure::rewrite(Def*, const Def* def) {
    if (auto join = def->isa<Join>()) {
        if (auto sigma = join->convert()) return sigma;
    } else if (auto meet = def->isa<Meet>()) {
        if (auto sigma = meet->convert()) return sigma;
    }

    return def;
}

}
