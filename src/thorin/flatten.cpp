#include "thorin/flatten.h"
#include "thorin/world.h"

#include <cassert>

namespace thorin {

static bool is_nominal(const Def* def) {
    return def->is_value() ? def->type()->isa_nominal() : def->isa_nominal();
}

static void flatten(std::vector<const Def*>& ops, const Def* def) {
    if (!is_nominal(def) && (def->isa<Tuple>() || def->isa<Pack>() || def->isa<Sigma>() || def->isa<Arr>())) {
        if (auto a = isa_lit_arity(def->arity())) {
            for (size_t i = 0; i != a; ++i) flatten(ops, proj(def, i));
        }
    }

    ops.emplace_back(def);
}

const Def* flatten(const Def* def) {
    std::vector<const Def*> ops;
    flatten(ops, def);
    auto res = def->is_value() ? def->world().tuple(ops, def->debug()) : def->world().sigma(ops, def->debug());
    res->dump();
    return res;
}

const Def* unflatten(const Def* def, const Def* type) {
    if (auto tuple = def->isa<Tuple>())
        return unflatten(tuple->ops(), type);
    if (type->isa<Sigma>() || (type->isa<Arr>() && type->as<Arr>()->arity()->isa<Lit>())) {
        Array<const Def*> ops(def->type()->lit_arity(), [&] (size_t i) {
            return def->out(i);
        });
        return unflatten(ops, type);
    }
    return def;
}

static const Def* unflatten(Defs defs, const Def* type, size_t& j) {
    auto& world = type->world();
    if (auto sigma = type->isa<Sigma>()) {
        Array<const Def*> ops(sigma->num_ops(), [&] (size_t i) {
            return unflatten(defs, sigma->op(i), j);
        });
        return world.tuple(ops);
    } else if (auto arr = type->isa<Arr>()) {
        if (auto lit = arr->arity()->isa<Lit>()) {
            Array<const Def*> ops(lit->get<nat_t>(), [&] (size_t) {
                return unflatten(defs, arr->codomain(), j);
            });
            return world.tuple(ops);
        }
    }
    return defs[j++];
}

const Def* unflatten(Defs defs, const Def* type) {
    size_t j = 0;
    auto def = unflatten(defs, type, j);
    assert(j == defs.size());
    return def;
}

}
