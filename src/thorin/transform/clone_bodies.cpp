#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

// TODO merge this with lift_builtins
void clone_bodies(World& world) {
    std::vector<Continuation*> todo;

    // TODO this looks broken: I guess we should do that in post-order as in lift_builtins
    for (auto continuation : world.copy_continuations()) {
        if (continuation->is_passed_to_accelerator())
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
                if (auto ucontinuation = use->isa_continuation())
                    ucontinuation->update_op(use.index(), ncontinuation);
                else {
                    auto primop = use->as<PrimOp>();
                    Array<const Def*> nops(primop->num_ops());
                    std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
                    nops[use.index()] = ncontinuation;
                    primop->replace(primop->rebuild(nops));
                }
            }
        }
    }
}

}
