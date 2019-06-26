#include "thorin/world.h"
#include "thorin/util.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

// TODO merge this with lift_builtins
void clone_bodies(World& world) {
    std::vector<Lam*> todo;

    // TODO this looks broken: I guess we should do that in post-order as in lift_builtins
    for (auto lam : world.copy_lams()) {
        if (is_passed_to_accelerator(lam))
            todo.push_back(lam);
    }

    for (auto lam : todo) {
        Scope scope(lam);
        bool first = true;
        for (auto use : lam->copy_uses()) {
            if (first) {
                first = false; // re-use the initial lam as first clone
            } else {
                // TODO this needs porting to new App stuff
                auto nlam = clone(scope);
                if (auto ulam = use->isa_nominal<Lam>())
                    ulam->set(use.index(), nlam);
                else {
                    auto primop = use->as<PrimOp>();
                    Array<const Def*> nops(primop->num_ops());
                    std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
                    nops[use.index()] = nlam;
                    primop->replace(primop->rebuild(world, primop->type(), nops, primop->name()));
                }
            }
        }
    }
}

}
