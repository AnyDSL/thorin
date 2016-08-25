#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

#define THRESHOLD 10

namespace thorin {

void inliner(World& world) {
    Scope::for_each(world, [] (const Scope& scope) {
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (auto callee = continuation->callee()->isa_continuation()) {
                if (!callee->empty() && callee->num_uses() <= 1 && !scope.contains(callee)) {
                    Scope callee_scope(callee);
                    continuation->jump(drop(callee_scope, continuation->args()), {}, continuation->jump_loc());
                }
            }
        }
    });

    Scope::for_each(world, [] (const Scope& scope) {
        if (scope.defs().size() < THRESHOLD)
            for (const auto& use : scope.entry()->uses())
                if (auto ucontinuation = use->isa_continuation())
                    if (use.index() == 0)
                        ucontinuation->jump(drop(scope, ucontinuation->args()), {}, ucontinuation->jump_loc());
    });

    debug_verify(world);
}

}
