#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/verify.h"
#include "anydsl2/transform/mangle.h"

namespace anydsl2 {

void inliner(World& world) {
    for (auto top : top_level_lambdas(world)) {
        if (!top->empty() && top->num_uses() <= 2) {
            for (auto use : top->uses()) {
                if (Lambda* ulambda = use->isa_lambda()) {
                    if (ulambda->to() == top) {
                        Scope scope(top);
                        if (!scope.contains(ulambda))
                            ulambda->jump(drop(scope, ulambda->args()), {});
                    }
                }
            }
        }
    }

    debug_verify(world);
}

} // namespace anydsl2
