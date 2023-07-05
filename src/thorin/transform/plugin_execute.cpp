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

        for (auto def : world.defs()) {
            auto cont = def->isa_nom<Continuation>();
            if (!cont) continue;

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

                auto app = use.def()->as<App>();
                assert(app->callee() == cont);

                if (app->num_uses() == 0) {
                    continue;
                }

                const Def* output = plugin_function(&world, app);
                const Def* app_rebuild = nullptr;
                if (output) {
                    app_rebuild = app->rebuild(world, world.bottom_type(), {app->arg(app->num_args() - 1), app->arg(0), output});
                } else {
                    app_rebuild = app->rebuild(world, world.bottom_type(), {app->arg(app->num_args() - 1), app->arg(0)});
                }
                app->replace_uses(app_rebuild);

                //partial_evaluation(world); //TODO: Some form of cleanup would be advisable here.
                evaluated = true;
            }

            if (evaluated)
                break;
        }

        world.cleanup();
    }

    world.mark_pe_done(false);
    world.cleanup();

    world.VLOG("end plugin_execute");
}

}
