#include "thorin/util.h"

#include "thorin/world.h"

namespace thorin {

bool is_memop(const Def* def) { return def->isa<App>() && def->out(0)->type()->isa<Mem>(); }

bool is_unit(const Def* def) {
    return def->type() == def->world().sigma();
}

std::tuple<const Axiom*, u16> get_axiom(const Def* def) {
    if (auto axiom = def->isa<Axiom>()) return {axiom, axiom->currying_depth()};
    if (auto app = def->isa<App>()) return {app->axiom(), app->currying_depth()};
    return {0, u16(-1)};
}

bool is_symmetric(const Def* def) {
    if (auto a = isa_lit_arity(def->type()->arity())) {
        if (auto z = proj<true>(def, *a, 0)) {
            if (auto b = isa_lit_arity(z->type()->arity())) {
                if (*a == *b) {
                    for (size_t i = 0; i != *a; ++i) {
                        for (size_t j = i+1; j != *a; ++j) {
                            auto ij = proj<true>(proj<true>(def, *a, i), *a, j);
                            auto ji = proj<true>(proj<true>(def, *a, j), *a, i);
                            if (ij == nullptr || ji == nullptr || ij != ji) return false;
                        }
                    }
                    return true;
                }
            }
        }
    }
    return false;
}

template<bool no_extract>
const Def* proj(const Def* def, u64 a, u64 i) {
    if (a == 1) return def;
    if (def == nullptr) return nullptr; // pass through nullptr for nested proj calls
    if (def->isa<Tuple>() || def->isa<Sigma>()) return def->op(i);
    if (auto pack = def->isa<Pack>()) { assert(i < pack->type()->lit_arity()); return pack->body();     }
    if (auto arr  = def->isa<Arr >()) { assert(i < arr         ->lit_arity()); return arr ->codomain(); }
    if (!no_extract && def->is_value()) { return def->world().extract(def, a, i); }
    return nullptr;
}

template<bool no_extract>
const Def* proj(const Def* def, u64 i) { return proj(def, def->lit_tuple_arity(), i); }

template const Def* proj<true >(const Def*, u64);
template const Def* proj<false>(const Def*, u64);
template const Def* proj<true >(const Def*, u64, u64);
template const Def* proj<false>(const Def*, u64, u64);

// TODO remove
Lam* get_param_lam(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->tuple()->as<Param>()->nominal()->as<Lam>();
    return def->as<Param>()->nominal()->as<Lam>();
}

// TODO remove
size_t get_param_index(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return as_lit<size_t>(extract->index());
    assert(def->isa<Param>());
    return 0;
}

std::vector<Peek> peek(const Def* param) {
    std::vector<Peek> peeks;
    size_t index = get_param_index(param);
    for (auto use : get_param_lam(param)->uses()) {
        if (auto app = use->isa<App>()) {
            for (auto use : app->uses()) {
                if (auto pred = use->isa_nominal<Lam>()) {
                    if (pred->body() == app)
                        peeks.emplace_back(app->arg(index), pred);
                }
            }
        }
    }

    return peeks;
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
    if (auto sigma = def->isa<Sigma>(); sigma && !sigma->isa_nominal())
        return def->world().sigma(merge(sigma->ops(), defs));
    return def->world().sigma(merge(def, defs));
}

const Def* merge_tuple(const Def* def, Defs defs) {
    auto& w = def->world();
    if (auto sigma = def->type()->isa<Sigma>(); sigma && !sigma->isa_nominal()) {
        Array<const Def*> tuple(sigma->num_ops(), [&](auto i) { return w.extract(def, i); });
        return w.tuple(merge(tuple, defs));
    }

    return def->world().tuple(merge(def, defs));
}

std::string tuple2str(const Def* def) {
    if (def == nullptr) return {};

    auto array = def->split(as_lit<nat_t>);
    return std::string(array.begin(), array.end());
}

}
