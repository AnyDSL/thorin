#include "thorin/flatten.h"
#include "thorin/world.h"

#include <cassert>

namespace thorin {

static bool is_sigma_or_arr(const Def* def) {
    return def->isa<Sigma>() || def->isa<Arr>();
}

static bool should_flatten(const Def* def) {
    return is_sigma_or_arr(def->is_value() ? def->type() : def);
}

static void flatten(std::vector<const Def*>& ops, const Def* def) {
    if (auto a = isa_lit<nat_t>(def->arity()); a && a != 1 && should_flatten(def)) {
        for (size_t i = 0; i != a; ++i)
            flatten(ops, proj(def, *a, i));
    } else {
        ops.emplace_back(def);
    }
}

const Def* flatten(const Def* def) {
    if (!should_flatten(def)) return def;
    std::vector<const Def*> ops;
    flatten(ops, def);
    return def->is_value() ? def->world().tuple(def->type(), ops, def->debug()) : def->world().sigma(ops, def->debug());
}

static const Def* unflatten(Defs defs, const Def* type, size_t& j) {
    if (auto a = isa_lit<nat_t>(type->arity()); a && a != 1) {
        auto& world = type->world();
        Array<const Def*> ops(*a, [&] (size_t i) { return unflatten(defs, proj(type, *a, i), j); });
        return world.tuple(type, ops);
    }

    return defs[j++];
}

const Def* unflatten(Defs defs, const Def* type) {
    size_t j = 0;
    auto def = unflatten(defs, type, j);
    assert(j == defs.size());
    return def;
}

const Def* unflatten(const Def* def, const Def* type) {
    return unflatten(def->split(), type);
}

}
