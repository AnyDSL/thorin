#include "thorin/pass/rw/type_erasure.h"

namespace thorin {

const Def* TypeErasure::rewrite(Def*, const Def* old_def, const Def* new_type, Defs new_ops, const Def* new_dbg) {
    if (auto vel = old_def->isa<Vel>()) {
        auto join = vel->type()->as<Join>();
        auto sigma = new_type->as<Sigma>();
        auto val = world().op_bitcast(sigma->op(1), new_ops[0], new_dbg);
        return world().tuple(sigma, {world().lit_int(join->num_ops(), join->find(vel->value()->type())), val});
    } else if (auto test = old_def->isa<Test>()) {
        test->value();
    }

    return old_def;
}

const Def* TypeErasure::rewrite(Def*, const Def* def) {
    return def;
}

const Sigma* TypeErasure::convert(const Join* join) {
    nat_t align = 0;
    nat_t size  = 0;

    for (auto op : join->ops()) {
        align = std::max(align, as_lit(world().op(Trait::align, op)));
        size  = std::max(size , as_lit(world().op(Trait::size , op)));
    }

    size = std::min(align, size);
    auto arr = world().arr(size, world().type_int_width(8));
    return world().sigma({world().type_int(join->num_ops()), arr})->as<Sigma>();
}

}
