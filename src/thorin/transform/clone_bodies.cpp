#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

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
        for (auto use : continuation->uses()) {
            if (first) {
                first = false; // re-use the initial continuation as first clone
            } else {
                auto ncontinuation = clone(scope);
                if (auto ucontinuation = use->isa_continuation())
                    ucontinuation->update_op(use.index(), ncontinuation);
                else {
                    auto primop = use->as<PrimOp>();
                    Array<const Def*> nops(primop->size());
                    std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
                    nops[use.index()] = ncontinuation;
                    primop->replace(primop->rebuild(nops));
                }
            }
        }
    }
}

}
