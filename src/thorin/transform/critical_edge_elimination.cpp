#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/top_level_scopes.h"

#include <iostream>

namespace thorin {

static Lambda* resolve(Lambda* dst, const char* suffix) {
    auto resolver = dst->stub(dst->name + suffix);
    resolver->jump(dst, resolver->params_as_defs());
    return resolver;
}

static void update_src(Lambda* src, Lambda* resolver, Lambda* dst) {
    World& world = src->world();
    Def nto;

    if (src->to()->isa_lambda()) {
        nto = resolver;
    } else if (auto select = src->to()->isa<Select>()) {
        if (select->tval() == dst)
            nto = world.select(select->cond(), resolver, select->fval());
        else {
            assert(select->fval() == dst);
            nto = world.select(select->cond(), select->tval(), resolver);
        }
    }

    if (nto)
        src->update_to(nto);
    else {
        for (size_t i = 0, e = src->num_args(); i != e; ++i) {
            if (src->arg(i) == dst) {
                src->update_arg(i, resolver);
                return;
            }
        }
        THORIN_UNREACHABLE;
    }
}

static void critical_edge_elimination(const Scope& scope) {
    // find critical edges
    std::vector<std::pair<Lambda*, Lambda*>> edges;
    for (auto lambda : scope) {
        const auto& preds = scope.preds(lambda);
        if (preds.size() > 1) {
            for (auto pred : preds) {
                if (scope.num_succs(pred) != 1)
                    edges.emplace_back(pred, lambda);
            }
        }
    }

    // remove them critical edges by inserting a resovling lambda
    for (auto edge : edges)
        update_src(edge.first, resolve(edge.second, ".crit"), edge.second);
}

void critical_edge_elimination(World& world) {
    // first we need to care about that this situation does not occur:
    //  a:                      b:
    //      A(..., c)               B(..., c)
    // such edges are not necessarily critical but we remove them here anyway as such situtions may cause trouble in some passes

    std::vector<Lambda*> todo;
    for (auto lambda : world.lambdas()) {
        if (lambda->is_basicblock()) {
            auto preds = lambda->preds();
            if (preds.size() > 1) {
                for (auto pred : preds) {
                    for (auto arg : pred->args()) {
                        if (arg == lambda) {
                            todo.push_back(lambda);
                            goto next_lambda;
                        }
                    }
                }
            }
        }
next_lambda:;
    }

    for (auto dst : todo) {
        for (auto src : dst->preds())
            update_src(src, resolve(dst, ".cascading"), dst);
    }

    for (auto scope : top_level_scopes(world))
        critical_edge_elimination(*scope);
}

}
