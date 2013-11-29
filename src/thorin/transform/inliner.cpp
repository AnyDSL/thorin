#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void inliner(World& world) {
    auto top_level = top_level_lambdas(world);
    for (auto top : top_level) {
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

} // namespace thorin
