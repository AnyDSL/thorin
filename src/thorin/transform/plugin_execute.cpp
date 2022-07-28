#include "thorin/world.h"
#include "thorin/transform/plugin_execute.h"
#include "thorin/analyses/scope.h"

namespace thorin {

void plugin_execute(World& world) {
    world.VLOG("start plugin_execute");

    for (auto cont : world.copy_continuations()) {
        if (cont->is_intrinsic() && cont->intrinsic() == Intrinsic::Plugin) {
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

                Continuation* y = world.continuation(world.fn_type({world.mem_type(), world.fn_type({world.mem_type()})}));
                y->jump(y->param(1), {y->param(0)});

                Continuation* x = world.continuation(world.fn_type({world.mem_type()}));
                x->jump(app->arg(app->num_args() - 1), {x->param(0), y});

                app->jump((Def*)output, {app->arg(0), x});
            }
        }
    }

    world.VLOG("end plugin_execute");
}

}
