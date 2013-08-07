#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/verify.h"
#include "anydsl2/transform/mangle.h"

namespace anydsl2 {

void inliner(World& world) {
    for (auto top : top_level_lambdas(world)) {
        if (top->num_uses() <= 2) {
            for (auto use : top->uses()) {
                if (use.index() == 0) {
                    if (Lambda* ulambda = use->isa_lambda()) {
                        Scope scope(top);
                        if (!scope.contains(ulambda))
                            ulambda->jump0(drop(scope, ulambda->args()));
                    }
                }
            }
        }
    }

    debug_verify(world);
}

} // namespace anydsl2
