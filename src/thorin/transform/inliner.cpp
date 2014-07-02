#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void inliner(World& world) {
    auto top = top_level_scopes(world);
    for (auto top_scope : top) {
        Scope scope(top_scope->entry()); // work around: the scopes in top may change so we recompute them here
        auto top = scope.entry();
        if (!top->empty() && top->num_uses() <= 2) {
            for (auto use : top->uses()) {
                if (auto ulambda = use->isa_lambda()) {
                    if (ulambda->to() == top) {
                        if (!scope.contains(ulambda))
                            ulambda->jump(drop(scope, ulambda->args()), std::initializer_list<Def>());
                    }
                }
            }
        }
    }

    debug_verify(world);
}

}
