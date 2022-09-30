#include "thorin/world.h"
#include "thorin/transform/plugin_execute.h"
#include "thorin/analyses/scope.h"

#include <vector>

namespace thorin {

void plugin_execute(World& world) {
    world.VLOG("start plugin_execute");

    std::vector<const Continuation*> plugin_intrinsics;

    for (auto cont : world.copy_continuations()) {
        if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Plugin) {
            plugin_intrinsics.push_back(cont);
            if (cont->attributes().depends) {
                cont->dump();
                cont->attributes().depends->dump();
            }
        }
    }

    sort(plugin_intrinsics.begin(), plugin_intrinsics.end(), [&](const Continuation* a, const Continuation* b) {
            const Continuation* depends = a;
            while (depends) {
              if (depends == b) return false;
              depends = depends->attributes().depends;
            }
            return true;
    });

    for (auto cont : plugin_intrinsics) {
        void * function_handle = world.search_plugin_function(cont->name());
        if (!function_handle) {
            world.ELOG("Plugin function not found for: {}", cont->name());
            continue;
        }
        auto plugin_function = (void*(*)(size_t, void**)) function_handle;

        for (auto use : cont->copy_uses()) {
            if (!use.def()->isa<App>()) {
                continue;
            }

            auto app = const_cast<App*>(use.def()->as<App>());

            Def* input_array[app->num_args() - 2];
            for (size_t i = 1, e = app->num_args() - 1; i < e; i++) {
                Def * input = const_cast<Def*>(app->arg(i));
                input_array[i - 1] = input;
            }

            void * output = plugin_function(app->num_args() - 2, (void **)input_array);
            app->jump(app->arg(app->num_args() - 1), {app->arg(0), (Def*)output});
        }
    }

    world.VLOG("end plugin_execute");
}

}
