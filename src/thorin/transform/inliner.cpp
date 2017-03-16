#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

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
    static const int factor = 4;
    static const int offset = 4;
    Scope::for_each(world, [] (Scope& scope) {
        if (scope.defs().size() < scope.entry()->num_params() * factor + offset) {
            for (const auto& use : scope.entry()->copy_uses()) {
                if (auto ucontinuation = use->isa_continuation()) {
                    if (use.index() == 0) {
                        ucontinuation->jump(drop(scope, ucontinuation->args()), {}, ucontinuation->jump_debug());
                        if (scope.contains(ucontinuation))
                            scope.update();
                    }
                }
            }
        }
    });

    debug_verify(world);
}

}
