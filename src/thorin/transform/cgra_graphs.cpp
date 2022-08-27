#include "thorin/world.h"
#include "thorin/continuation.h"
#include "thorin/transform/cgra_graphs.h"
#include "thorin/transform/hls_channels.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/verify.h"
#include "thorin/type.h"

namespace thorin {

void cgra_graphs(Importer& importer) {

    auto& world = importer.world();
    //world.dump();

    Scope::for_each(world, [&] (Scope& scope) {
        Def2Mode def2mode;
        extract_kernel_channels(schedule(scope), def2mode);
        for (auto elem : def2mode)
                elem.first->dump();
    });

    world.cleanup();


}

}
