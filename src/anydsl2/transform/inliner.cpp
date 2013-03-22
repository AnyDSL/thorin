#include "anydsl2/transform/inliner.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

void inliner(World& world) {
    for_all (top, find_root_lambdas(world)) {
        if (top->num_uses() <= 2) {
            for_all (use, top->uses()) {
                if (use.index() == 0) {
                    if (Lambda* ulambda = use->isa_lambda()) {
                        Scope scope(top);
                        if (!scope.contains(ulambda))
                            ulambda->jump0(scope.drop(ulambda->args()));
                    }
                }
            }
        }
    }
}

} // namespace anydsl2
