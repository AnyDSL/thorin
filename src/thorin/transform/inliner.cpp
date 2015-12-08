#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void inliner(World& world) {
    Scope::for_each(world, [] (const Scope& scope) {
        auto top = scope.entry();
        if (!top->empty() && top->num_uses() <= 2) {
            for (auto use : top->uses()) {
                if (auto ulambda = use->isa_lambda()) {
                    if (ulambda->to() == top) {
                        if (!scope.outer_contains(ulambda))
                            ulambda->jump({}, drop(scope, ulambda->args()), {}); // TODO use type_args here instead of empty Type2Type set
                    }
                }
            }
        }
    });

    debug_verify(world);
}

}
