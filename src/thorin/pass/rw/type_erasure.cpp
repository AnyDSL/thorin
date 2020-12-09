#include "thorin/pass/rw/type_erasure.h"

namespace thorin {

const Def* TypeErasure::rewrite(Def*, const Def* def) {
    if (auto vel = def->isa<Vel>()) {
        auto join = vel->type()->as<Join>();
        auto sigma = convert(join);
        auto val = world().op_bitcast(sigma->op(1), vel->value());
        return world().tuple(sigma, {world().lit_int(join->num_ops(), join->find(vel->value()->type())), val});
    } else if (auto test = def->isa<Test>()) {
        test->value();
    }

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
