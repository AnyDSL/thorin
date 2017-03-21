#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"

namespace thorin {

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
