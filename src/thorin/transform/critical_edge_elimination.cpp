#include "thorin/lambda.h"
#include "thorin/world.h"

namespace thorin {

void critical_edge_elimination(World& world) {
    // first we need to care about that this situation does not occur:
    //  a:                      b:
    //      A(..., c)               B(..., c)
    // such edges are not necessarily critical but we remove them here anyway

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

    for (auto lambda : todo) {
        for (auto pred : lambda->preds()) {
            // create new lambda
            Type2Type map;
            auto resolver = lambda->stub(map, lambda->name + ".cascading");
            resolver->jump(lambda, resolver->params_as_defs());

            // update pred
            for (size_t i = 0, e = pred->num_args(); i != e; ++i) {
                if (pred->arg(i) == lambda) {
                    pred->update_arg(i, resolver);
                    goto next_pred;
                }
            }
            THORIN_UNREACHABLE;
next_pred:;
        }
    }

    // now remove critical edges
    // TODO
}

}
