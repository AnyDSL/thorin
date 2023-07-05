#include "thorin/world.h"
#include "thorin/transform/plugin_execute.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/analyses/scope.h"

namespace thorin {

void plugin_execute(World& world, std::vector<World::PluginIntrinsic> intrinsics) {
    world.VLOG("start plugin_execute");

    // assume that dependents precede dependencies
    while (!intrinsics.empty()) {
        intrinsics.back().impl->transform(&world, intrinsics.back().cont);
        intrinsics.pop_back();
    }

    world.mark_pe_done(false);
    world.cleanup();

    world.VLOG("end plugin_execute");
}

}
