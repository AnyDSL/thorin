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
            auto continuation = n->continuation();
            if (auto callee = continuation->callee()->isa_continuation()) {
                if (!callee->empty() && callee->num_uses() <= 1 && !scope.contains(callee)) {
                    Scope to_scope(callee);
                    continuation->jump(drop(to_scope, continuation->type_args(), continuation->args()), {}, {}, continuation->jump_loc());
                }
            }
        }
    });

    debug_verify(world);
}

}
