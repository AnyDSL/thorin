#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/log.h"

namespace thorin {

static Lambda* resolve(Lambda* dst, const char* suffix) {
    auto resolver = dst->stub(dst->name + suffix);
    resolver->jump(dst, resolver->params_as_defs());
    return resolver;
}

static void update_src(Lambda* src, Lambda* resolver, Lambda* dst) {
    if (src->to() == dst)
        src->update_to(resolver);
    else if (src->to() == src->world().branch()) {
        if (src->arg(1) == dst)
            src->branch(src->arg(0), resolver, src->arg(2));
        else {
            assert(src->arg(2) == dst);
            src->branch(src->arg(0), src->arg(1), resolver);
        }
    } else {
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
    auto& cfg = scope.f_cfg();
    // find critical edges
    std::vector<std::pair<Lambda*, Lambda*>> edges;
    for (auto lambda : scope) {
        if (!lambda->to()->isa<Bottom>()) {
            const auto& preds = cfg.preds(cfg[lambda]);
            if (preds.size() > 1) {
                for (auto pred : preds) {
                    auto lpred = pred->lambda();
                    if (cfg.num_succs(pred) != 1) {
                        WLOG("critical edge: %, %", lpred->unique_name(), lambda->unique_name());
                        edges.emplace_back(lpred, lambda);
                    }
                }
            }
        }
    }

    return;

    // remove critical edges by inserting a resovling lambda
    for (auto edge : edges) {
        auto src = edge.first;
        auto dst = edge.second;
        update_src(src, resolve(dst, ".crit"), dst);
    }
}

void critical_edge_elimination(World& world) {
    // first we need to care about that this situation does not occur:
    //  a:                      b:
    //      A(..., c)               B(..., c)
    // such edges are not really critical but we remove them here anyway as such situtions may cause trouble in some passes

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

    Scope::for_each(world, [] (const Scope& scope) { critical_edge_elimination(scope); });
    debug_verify(world);
}

}
