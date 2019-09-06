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

std::tuple<const Axiom*, u16> get_axiom(const Def* def) {
    if (auto axiom = def->isa<Axiom>()) return {axiom, axiom->currying_depth()};
    if (auto app = def->isa<App>()) return {app->axiom(), app->currying_depth()};
    return {0, u16(-1)};
}

void app_to_dropped_app(Lam* src, Lam* dst, const App* app) {
    std::vector<const Def*> nargs;
    auto src_app = src->body()->as<App>();
    for (size_t i = 0, e = src_app->num_args(); i != e; ++i) {
        if (app->arg(i)->isa<Top>())
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

    std::string result;
    for (size_t i = 0, e = def->lit_arity(); i != e; ++i)
        result.push_back(as_lit<u64>(def->out(i)));
    return result;
}

}
