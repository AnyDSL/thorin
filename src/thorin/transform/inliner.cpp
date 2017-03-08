#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

#define THRESHOLD 10

namespace thorin {

void force_inline(Scope& scope, int threshold) {
    for (bool todo = true; todo && threshold-- != 0;) {
        todo = false;
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (auto callee = continuation->callee()->isa_continuation()) {
                if (!callee->empty() && !scope.contains(callee)) {
                    Scope callee_scope(callee);
                    continuation->jump(drop(callee_scope, continuation->args()), {}, continuation->jump_debug());
                    todo = true;
                }
            }
        }

        if (todo)
            scope.update();
    }

    for (auto n : scope.f_cfg().reverse_post_order()) {
        auto continuation = n->continuation();
        if (auto callee = continuation->callee()->isa_continuation()) {
            if (!callee->empty() && !scope.contains(callee)) {
                WLOG(callee, "couldn't inline {} at {}", scope.entry(), continuation->jump_location());
            }
        }
    }
}

void inliner(World& world) {
    Scope::for_each(world, [] (const Scope& scope) {
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (auto callee = continuation->callee()->isa_continuation()) {
                if (!callee->empty() && callee->num_uses() <= 1 && !scope.contains(callee)) {
                    Scope callee_scope(callee);
                    continuation->jump(drop(callee_scope, continuation->args()), {}, continuation->jump_debug());
                }
            }
        }
    });

    Scope::for_each(world, [] (const Scope& scope) {
        if (scope.defs().size() < THRESHOLD)
            for (const auto& use : scope.entry()->copy_uses())
                if (auto ucontinuation = use->isa_continuation())
                    if (use.index() == 0)
                        ucontinuation->jump(drop(scope, ucontinuation->args()), {}, ucontinuation->jump_debug());
    });

    debug_verify(world);
}

}
