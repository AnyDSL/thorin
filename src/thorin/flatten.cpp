#include "thorin/flatten.h"
#include "thorin/world.h"

#include <cassert>

namespace thorin {

const Def* Flattener::flatten(const Def* def) {
    if (auto new_def = old2new.lookup(def))
        return *new_def;
    auto& world = def->world();
    if (auto pack = def->isa<Pack>()) {
        if (!pack->arity()->isa<Lit>()) return old2new[def] = def;
        auto body = flatten(pack->body());
        auto n = as_lit<nat_t>(pack->arity());
        if (body->type()->arity()->isa<Lit>()) {
            auto m = body->type()->lit_arity();
            Array<const Def*> ops(n * m);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j)
                    ops[i * m + j] = body->out(j);
            }
            return old2new[def] = world.tuple(ops);
        } else {
            return old2new[def] = def;
        }
    } else if (auto tuple = def->isa<Tuple>()) {
        std::vector<const Def*> ops;
        for (auto op : tuple->ops()) {
            auto flat_op = flatten(op);
            if (auto lit_arity = flat_op->type()->arity()->isa<Lit>()) {
                for (size_t i = 0, n = lit_arity->get<nat_t>(); i < n; ++i)
                    ops.push_back(flat_op->out(i));
            } else {
                ops.push_back(flat_op);
            }
        }
        return old2new[def] = world.tuple(ops);
    } else if (auto arr = def->isa<Arr>()) {
        if (!arr->domain()->isa<Lit>()) return old2new[def] = def;
        auto codomain = flatten(arr->codomain());
        auto n = as_lit<nat_t>(arr->domain());
        if (codomain->arity()->isa<Lit>()) {
            auto m = codomain->lit_arity();
            Array<const Def*> ops(n * m);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j)
                    ops[i * m + j] = codomain->isa<Arr>() ? codomain->as<Arr>()->codomain() : codomain->op(j);
            }
            return old2new[def] = world.sigma(ops);
        } else {
            return old2new[def] = def;
        }
    } else if (auto sigma = def->isa<Sigma>()) {
        std::vector<const Def*> ops;
        for (auto op : sigma->ops()) {
            auto flat_op = flatten(op);
            if (auto lit_arity = flat_op->arity()->isa<Lit>()) {
                for (size_t i = 0, n = lit_arity->get<nat_t>(); i < n; ++i)
                    ops.push_back(flat_op->isa<Arr>() ? flat_op->as<Arr>()->codomain() : flat_op->op(i));
            } else {
                ops.push_back(flat_op);
            }
        }
        return old2new[def] = world.sigma(ops);
    } else {
        return old2new[def] = def;
    }
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
