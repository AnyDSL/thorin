#ifndef THORIN_PLUGIN_HELPERS_H
#define THORIN_PLUGIN_HELPERS_H

#include "plugin.h"
#include "continuation.h"
#include "world.h"

namespace thorin {
    template <typename Plugin>
    class plugin_intrinsic_app_wrapper : public virtual plugin_intrinsic {
        virtual const Def* transform(World* world, const App* app) = 0;

        void transform(World* world, const Continuation* cont) final override {
            for (auto use : cont->copy_uses()) {
                if (!use.def()->isa<App>())
                    continue;

                auto app = use.def()->as<App>();

                // transform plugin(mem, args, ret) to ret(mem, func(args))

                auto output = static_cast<Plugin*>(this)->transform(world, app);

                auto app_rebuild =
                    output ? app->rebuild(*world, world->bottom_type(), {app->arg(app->num_args() - 1), app->arg(0), output})
                           : app->rebuild(*world, world->bottom_type(), {app->arg(app->num_args() - 1), app->arg(0)});

                app->replace_uses(app_rebuild);
            }
        }

    protected:
        plugin_intrinsic_app_wrapper() = default;
        plugin_intrinsic_app_wrapper(const plugin_intrinsic_app_wrapper&) = default;
        plugin_intrinsic_app_wrapper(plugin_intrinsic_app_wrapper&&) = default;
        plugin_intrinsic_app_wrapper& operator =(const plugin_intrinsic_app_wrapper&) = default;
        plugin_intrinsic_app_wrapper& operator =(plugin_intrinsic_app_wrapper&&) = default;
        ~plugin_intrinsic_app_wrapper() = default;
    };
}

#endif
