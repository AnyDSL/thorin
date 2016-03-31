#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void clone_bodies(World& world) {
    std::vector<Lambda*> todo;

    // TODO this looks broken: I guess we should do that in post-order as in lift_builtins
    for (auto lambda : world.copy_lambdas()) {
        if (lambda->is_passed_to_accelerator())
            todo.push_back(lambda);
    }

    for (auto lambda : todo) {
        Scope scope(lambda);
        bool first = true;
        for (auto use : lambda->uses()) {
            if (first) {
                first = false; // re-use the initial lambda as first clone
            } else {
                auto nlambda = clone(scope);
                if (auto ulambda = use->isa_lambda())
                    ulambda->update_op(use.index(), nlambda);
                else {
                    auto primop = use->as<PrimOp>();
                    Array<const Def*> nops(primop->size());
                    std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
                    nops[use.index()] = nlambda;
                    primop->replace(primop->rebuild(nops));
                }
            }
        }
    }
}

}
