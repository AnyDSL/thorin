#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

namespace thorin {

void inliner(World& world) {
    Scope::for_each(world, [] (const Scope& scope) {
        for (auto n : scope.f_cfg().post_order()) {
            auto lambda = n->lambda();
            if (auto to_lambda = lambda->to()->isa_lambda()) {
                if (!to_lambda->empty() && to_lambda->num_uses() <= 1 && !scope.contains(to_lambda)) {
                    Scope to_scope(to_lambda);
                    lambda->jump(drop(to_scope, lambda->args()), {}, lambda->jump_loc());
                }
            }
        }
    });

    debug_verify(world);
}

}
