#include "thorin/util.h"

#include "thorin/world.h"

namespace thorin {

bool is_unit(const Def* def) {
    return def->type() == def->world().sigma();
}

bool is_const(const Def* def) {
    unique_stack<DefSet> stack;
    stack.push(def);

    while (!stack.empty()) {
        auto def = stack.pop();
        if (def->isa<Param>()) return false;
        if (def->isa<Hlt>()) return false;
        if (!def->isa_nominal()) {
            for (auto op : def->ops())
                stack.push(op);
        }
        // lams are always const
    }

    return true;
}

bool is_primlit(const Def* def, int64_t val) {
    if (auto lit = def->isa<Lit>()) {
        if (auto prim_type = lit->type()->isa<PrimType>()) {
            switch (prim_type->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit->box().get_##T() == T(val);
#include "thorin/tables/primtypetable.h"
                case PrimType_bool: return lit->box().get_bool() == bool(val);
                default: ; // FALLTHROUGH
            }
        }
    }

    return false;
}

bool is_minus_zero(const Def* def) {
    if (auto lit = def->isa<Lit>()) {
        if (auto prim_type = lit->type()->isa<PrimType>()) {
            switch (prim_type->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit->box().get_##M() == M(0);
#define THORIN_F_TYPE(T, M) case PrimType_##T: return lit->box().get_##M() == M(-0.0);
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
        }
    }
    return false;
}

void app_to_dropped_app(Lam* src, Lam* dst, const App* app) {
    std::vector<const Def*> nargs;
    auto src_app = src->body()->as<App>();
    for (size_t i = 0, e = src_app->num_args(); i != e; ++i) {
        if (is_top(app->arg(i)))
            nargs.push_back(src_app->arg(i));
    }

    src->app(dst, nargs, src_app->debug());
}

// TODO remove
Lam* get_param_lam(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->agg()->as<Param>()->lam();
    return def->as<Param>()->lam();
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

bool visit_uses(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    if (!lam->is_intrinsic()) {
        for (auto use : lam->uses()) {
            auto def = include_globals && use->isa<Global>() ? use->uses().begin()->def() : use.def();
            if (auto lam = def->isa_nominal<Lam>())
                if (func(lam))
                    return true;
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    return visit_uses(lam, [&] (Lam* lam) {
        if (auto callee = lam->app()->callee()->isa_nominal<Lam>())
            return callee->is_intrinsic() && func(callee);
        return false;
    }, include_globals);
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

Array<const Def*> merge_tuple_or_sigma(const Def* tuple_or_sigma, Defs defs) {
    Array<const Def*> result(tuple_or_sigma->num_ops() + defs.size());
    auto i = std::copy(tuple_or_sigma->ops().begin(), tuple_or_sigma->ops().end(), result.begin());
    std::copy(defs.begin(), defs.end(), i);
    return result;
}

const Def* merge_sigma(const Def* def, Defs defs) {
    if (auto sigma = def->isa<Sigma>(); sigma && !sigma->isa_nominal())
        return def->world().sigma(merge_tuple_or_sigma(sigma, defs));
    return def->world().sigma(merge(def, defs));
}

const Def* merge_tuple(const Def* def, Defs defs) {
    if (auto tuple = def->isa<Tuple>(); tuple && !tuple->type()->isa_nominal())
        return def->world().tuple(merge_tuple_or_sigma(tuple, defs));
    return def->world().tuple(merge(def, defs));
}

}
