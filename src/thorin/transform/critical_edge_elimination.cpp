#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/log.h"

namespace thorin {

static bool update_src(Continuation* src, Continuation* dst, const char* suffix) {
    auto resolve = [&] (Continuation* dst) {
        auto resolver = dst->stub();
        dst->debug().set(dst->name() + suffix);
        resolver->jump(dst, resolver->params_as_defs(), resolver->jump_debug());
        return resolver;
    };

    for (size_t i = 0, e = src->num_ops(); i != e; ++i) {
        if (src->op(i) == dst) {
            src->update_op(i, resolve(dst));
            return true;
        }
    }

    DLOG("cannot remove critical edge {} -> {}", src, dst);
    return false;
}

static void critical_edge_elimination(Scope& scope) {
    bool dirty = false;
    const auto& cfg = scope.f_cfg();
    for (auto n : cfg.post_order()) {
        if (cfg.num_preds(n) > 1 && n->continuation()->order() == 1) {
            for (auto pred : cfg.preds(n)) {
                if (cfg.num_succs(pred) != 1) {
                    DLOG("critical edge: {} -> {}", pred, n);
                    dirty |= update_src(pred->continuation(), n->continuation(), "_crit");
                }
            }
        }
    }

    if (dirty)
        scope.update();
}

void critical_edge_elimination(World& world) {
    // first we need to care about that this situation does not occur:
    //  a:                      b:
    //      A(..., c)               B(..., c)
    // such edges are not really critical but we remove them here anyway as such situtions may cause trouble in some passes

    std::vector<Continuation*> todo;
    for (auto continuation : world.continuations()) {
        if (continuation->is_basicblock()) {
            auto preds = continuation->preds();
            if (preds.size() > 1) {
                for (auto pred : preds) {
                    for (auto arg : pred->args()) {
                        if (arg == continuation) {
                            todo.push_back(continuation);
                            goto next_continuation;
                        }
                    }
                }
            }
        }
next_continuation:;
    }

    for (auto dst : todo) {
        for (auto src : dst->preds())
            update_src(src, dst, "_cascading");
    }

    Scope::for_each(world, [] (Scope& scope) { critical_edge_elimination(scope); });
    debug_verify(world);
}

}
