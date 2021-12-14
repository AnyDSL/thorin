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

static bool nom_val_or_typ(const Def *def) {
    auto typ = (def->sort() == Sort::Term) ? def->type() : def;
    return typ->isa_nom();
}

size_t flatten(DefVec& ops, const Def* def, bool flatten_noms) {
    if (auto a = isa_lit<nat_t>(def->arity()); a && *a != 1 && should_flatten(def)
            && flatten_noms == nom_val_or_typ(def)) {
        auto n = 0;
        for (size_t i = 0; i != *a; ++i)
            n += flatten(ops, proj(def, *a, i), flatten_noms);
        return n;
    } else {
        ops.emplace_back(def);
        return 1;
    }
}

const Def* flatten(const Def* def) {
    if (!should_flatten(def)) return def;
    DefVec ops;
    flatten(ops, def);
    return def->sort() == Sort::Term ? def->world().tuple(def->type(), ops, def->dbg()) : def->world().sigma(ops, def->dbg());
}

static const Def* unflatten(Defs defs, const Def* type, size_t& j) {
    if (!defs.empty() && defs[0]->type() == type)
        return defs[j++];
    if (auto a = isa_lit<nat_t>(type->arity()); a && *a != 1) {
        auto& world = type->world();
        DefArray ops(*a, [&] (size_t i) { return unflatten(defs, proj(type, *a, i), j); });
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

DefArray merge(const Def* def, Defs defs) {
    return DefArray(defs.size() + 1, [&](auto i) { return i == 0 ? def : defs[i-1]; });
}

DefArray merge(Defs a, Defs b) {
    DefArray result(a.size() + b.size());
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
        DefArray tuple(a, [&](auto i) { return w.extract(def, a, i); });
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
