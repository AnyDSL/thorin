#include "thorin/world.h"
#include "thorin/transform/plugin_execute.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/analyses/scope.h"

#include <vector>

namespace thorin {

void plugin_execute(Thorin& thorin) {
    World& world = thorin.world();
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
            void * function_handle = thorin.search_plugin_function(cont->name());
            if (!function_handle) {
                world.ELOG("Plugin function not found for: {}", cont->name());
                continue;
            }
            auto plugin_function = (void*(*)(size_t, void**)) function_handle;

            bool evaluated = false;
            for (auto use : cont->copy_uses()) {
                if (!use.def()->isa<App>()) {
                    continue;
                }

                auto app = const_cast<App*>(use.def()->as<App>());

                if (app->num_uses() == 0) {
                    continue;
                }

                Def* input_array[app->num_args() - 2];
                for (size_t i = 1, e = app->num_args() - 1; i < e; i++) {
                    Def * input = const_cast<Def*>(app->arg(i));
                    input_array[i - 1] = input;
                }

                void * output = plugin_function(app->num_args() - 2, (void **)input_array);
                app->jump(app->arg(app->num_args() - 1), {app->arg(0), (Def*)output});

                //partial_evaluation(world); //TODO: Some form of cleanup would be advisable here.
                evaluated = true;
            }

            if (evaluated)
                break;
        }

        thorin.cleanup(); //Warning: This must not change the world, there are still references to intrinsics being maintained here.
    }

    world.mark_pe_done(false);
    thorin.cleanup();

    world.VLOG("end plugin_execute");
}

}
