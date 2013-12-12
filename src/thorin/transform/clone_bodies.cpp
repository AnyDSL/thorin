#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void clone_bodies(World& world) {
    for (auto lambda : world.copy_lambdas()) {
        if (lambda->is_connected_to_builtin()) {
            Scope scope(lambda);
            bool first = true;
            for (auto use : lambda->uses()) {
                if (first) {
                    first = false; // re-use the initial lambda as first clone
                    continue;
                }

                auto nlambda = clone(scope);
                if (auto ulambda = use->isa_lambda())
                    ulambda->update_op(use.index(), nlambda);
                else {
                    auto primop = use->as<PrimOp>();
                    Array<Def> nops(primop->size());
                    std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
                    nops[use.index()] = nlambda;
                    primop->replace(world.rebuild(primop, nops));
                }
            }
        }
    }
}

}
