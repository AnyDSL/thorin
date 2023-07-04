#include "thorin/world.h"
#include "thorin/transform/plugin_execute.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/analyses/scope.h"

#include <vector>

namespace thorin {

void plugin_execute(World& world) {
    world.VLOG("start plugin_execute");

    std::vector<const Continuation*> plugin_intrinsics;

    while (true) {
        plugin_intrinsics.clear();

        for (auto cont : world.copy_continuations()) {
            if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Plugin) {
                plugin_intrinsics.push_back(cont);
            }
        }

        if (plugin_intrinsics.empty())
            break;

        sort(plugin_intrinsics.begin(), plugin_intrinsics.end(), [&](const Continuation* a, const Continuation* b) {
                const Continuation* depends = a;
                while (depends) {
                  if (depends == b) return false;
                  depends = depends->attributes().depends;
                }
                return true;
        });

        for (auto cont : plugin_intrinsics) {
            auto plugin_function = world.search_plugin_function(cont->name().c_str());
            if (!plugin_function) {
                world.ELOG("Plugin function not found for: {}", cont->name());
                continue;
            }

            bool evaluated = false;
            for (auto use : cont->copy_uses()) {
                if (!use.def()->isa<App>()) {
                    continue;
                }

                auto app = const_cast<App*>(use.def()->as<App>());

                if (app->num_uses() == 0) {
                    continue;
                }

                void* output = plugin_function(&world, app);
                if (output)
                    app->jump(app->arg(app->num_args() - 1), {app->arg(0), (Def*)output});
                else
                    app->jump(app->arg(app->num_args() - 1), {app->arg(0)});

                //partial_evaluation(world); //TODO: Some form of cleanup would be advisable here.
                evaluated = true;
            }

            if (evaluated)
                break;
        }

        world.cleanup(); //Warning: This must not change the world, there are still references to intrinsics being maintained here.
    }

    world.mark_pe_done(false);
    world.cleanup();

    world.VLOG("end plugin_execute");
}

}
