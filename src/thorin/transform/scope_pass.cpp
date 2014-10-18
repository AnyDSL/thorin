#if 0

#include "thorin/transform/scope_pass.h"
#include "thorin/analyses/verify.h"

namespace thorin {

void ScopePass::run(World& world) {
    top_level_scopes(world, [&] (const Scope& scope) { run(scope); });
    debug_verify(world);
}

}
#endif
