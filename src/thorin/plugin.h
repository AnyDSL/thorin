#ifndef THORIN_PLUGIN_H
#define THORIN_PLUGIN_H

#include <memory>

namespace thorin {
    class Def;
    class Continuation;
    class App;
    class World;

    struct plugin_intrinsic {
        virtual void transform(World* world, const Continuation* cont) = 0;
        virtual void destroy() = 0;

    protected:
        plugin_intrinsic() = default;
        plugin_intrinsic(const plugin_intrinsic&) = default;
        plugin_intrinsic(plugin_intrinsic&&) = default;
        plugin_intrinsic& operator =(const plugin_intrinsic&) = default;
        plugin_intrinsic& operator =(plugin_intrinsic&&) = default;
        ~plugin_intrinsic() = default;
    };

    struct plugin_deleter {
        void operator ()(plugin_intrinsic* ptr) const {
            ptr->destroy();
        }
    };

    using unique_plugin_intrinsic = std::unique_ptr<plugin_intrinsic, plugin_deleter>;

    extern "C" {
        using plugin_init_func_t = bool();
        using plugin_intrinsic_create_func_t = plugin_intrinsic*();
    }
}

#endif
