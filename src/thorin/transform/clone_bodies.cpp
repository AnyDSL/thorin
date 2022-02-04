#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

/*
 * TODO rewrite
 */

// TODO merge this with lift_builtins
void clone_bodies(World& /*world*/) {
#if 0
    std::vector<Lam*> todo;

    // TODO this looks broken: I guess we should do that in post-order as in lift_builtins
    for (auto continuation : world.copy_continuations()) {
        if (is_passed_to_accelerator(continuation))
            todo.push_back(continuation);
    }

    for (auto continuation : todo) {
        Scope scope(continuation);
        bool first = true;
        for (auto use : continuation->copy_uses()) {
            if (first) {
                first = false; // re-use the initial continuation as first clone
            } else {
                auto ncontinuation = clone(scope);
                if (auto uapp = use->isa<App>()) {
                    if (uapp->is_replaced()) continue; // dead app node
                    auto napp = uapp->with_different_op(use.index(), ncontinuation);
                    uapp->replace(napp);
                } else if (auto primop = use->isa<PrimOp>()) {
                    Array<const Def*> nops(primop->num_ops());
                    std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
                    nops[use.index()] = ncontinuation;
                    primop->replace(primop->rebuild(world, primop->type(), nops));
                }
            }
        }
    }
#endif
}

}
