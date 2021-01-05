#include "thorin/tuple.h"

#include "thorin/world.h"

#include <cassert>

namespace thorin {

const Def* proj(const Def* def, u64 a, u64 i, const Def* dbg) {
    auto& world = def->world();

    if (a == 1 && (!def->isa_nom<Sigma>() && !def->type()->isa_nom<Sigma>())) return def;
    if (def->isa<Tuple>() || def->isa<Sigma>()) return def->op(i);

    if (auto arr = def->isa<Arr>()) {
        if (arr->arity()->isa<Top>()) return arr->body();
        return arr->apply(world.lit_int(as_lit(arr->arity()), i)).back();
    }

    if (auto pack = def->isa<Pack>()) {
        if (pack->arity()->isa<Top>()) return pack->body();
        return pack->apply(world.lit_int(as_lit(pack->arity()), i)).back();
    }

    if (def->sort() == Sort::Term) { return def->world().extract(def, a, i, dbg); }

    return nullptr;
}

static bool should_flatten(const Def* def) {
    return is_sigma_or_arr(def->sort() == Sort::Term ? def->type() : def);
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
    return def->sort() == Sort::Term ? def->world().tuple(def->type(), ops, def->dbg()) : def->world().sigma(ops, def->dbg());
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
    return unflatten(def->split(as_lit(def->arity())), type);
}

bool is_unit(const Def* def) {
    return def->type() == def->world().sigma();
}

bool is_tuple_arg_of_app(const Def* def) {
    if (!def->isa<Tuple>()) return false;
    for (auto& use : def->uses()) {
        if (use.index() == 1 && use->isa<App>())
            continue;
        if (!is_tuple_arg_of_app(use.def()))
            return false;
    }
    return true;
}

Array<const Def*> merge(const Def* def, Defs defs) {
    return Array<const Def*>(defs.size() + 1, [&](auto i) { return i == 0 ? def : defs[i-1]; });
}

Array<const Def*> merge(Defs a, Defs b) {
    Array<const Def*> result(a.size() + b.size());
    auto i = std::copy(a.begin(), a.end(), result.begin());
    std::copy(b.begin(), b.end(), i);
    return result;
}

const Def* merge_sigma(const Def* def, Defs defs) {
    if (auto sigma = def->isa<Sigma>(); sigma && !sigma->isa_nom())
        return def->world().sigma(merge(sigma->ops(), defs));
    return def->world().sigma(merge(def, defs));
}

const Def* merge_tuple(const Def* def, Defs defs) {
    auto& w = def->world();
    if (auto sigma = def->type()->isa<Sigma>(); sigma && !sigma->isa_nom()) {
        auto a = sigma->num_ops();
        Array<const Def*> tuple(a, [&](auto i) { return w.extract(def, a, i); });
        return w.tuple(merge(tuple, defs));
    }

    return def->world().tuple(merge(def, defs));
}

std::string tuple2str(const Def* def) {
    if (def == nullptr) return {};

    auto array = def->split(as_lit(def->arity()), as_lit<nat_t>);
    return std::string(array.begin(), array.end());
}

}
