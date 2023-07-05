#ifndef THORIN_TRANSFORM_PLUGIN_EXECUTE_H
#define THORIN_TRANSFORM_PLUGIN_EXECUTE_H

#include "thorin/world.h"

namespace thorin {

void plugin_execute(World&, std::vector<World::PluginIntrinsic> intrinsics);

}

#endif
