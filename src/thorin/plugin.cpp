#include "thorin/world.h"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <execinfo.h>
#include <dlfcn.h>
#endif

namespace thorin {
[[nodiscard]] static void* load_plugin_module(const char* plugin_name) {
#ifdef _WIN32
    return LoadLibraryA(plugin_name);
#else
    return dlopen(plugin_name, RTLD_LAZY | RTLD_GLOBAL);
#endif
}

static bool unload_plugin_module(void* plugin_module) {
#ifdef _WIN32
    return FreeLibrary(static_cast<HMODULE>(plugin_module)) == TRUE;
#else
    return dlclose(plugin_module) == 0;
#endif
}

template <typename T>
[[nodiscard]] static T* lookup_plugin_function(void* plugin_module, const char* function_name) {
#ifdef _WIN32
    return reinterpret_cast<T*>(GetProcAddress(static_cast<HMODULE>(plugin_module), function_name));
#else
    return reinterpret_cast<T*>(dlsym(plugin_module, function_name));
#endif
}

bool World::load_plugin(const char* plugin_name) {
    void* module = load_plugin_module(plugin_name);

    if (!module) {
        ELOG("failed to load plugin {}", plugin_name);
        return false;
    }

    if (auto init = lookup_plugin_function<plugin_init_func_t>(module, "init")) {
        if (!init()) {
            ELOG("failed to initialize plugin {}", plugin_name);
            unload_plugin_module(module);
            return false;
        }
    } else {
        ILOG("plugin {} did not supply an init function", plugin_name);
    }

    data_.plugin_modules_.push_back(module);
    return true;
}

unique_plugin_intrinsic World::load_plugin_intrinsic(const char* function_name) const {
    for (auto plugin_module : data_.plugin_modules_) {
        if (auto create_intrinsic = lookup_plugin_function<plugin_intrinsic_create_func_t>(plugin_module, function_name))
            return unique_plugin_intrinsic(create_intrinsic());
    }

    return nullptr;
}

void World::link_plugin_intrinsic(Continuation* cont) {
    auto intrinsic = load_plugin_intrinsic(cont->name().c_str());

    if (!intrinsic)
        error(cont->loc(), "failed to link plugin intrinsic {}", cont->name());

    cont->attributes().intrinsic = Intrinsic::Plugin;
    data_.plugin_intrinsics_.push_back({ cont, std::move(intrinsic) });
}
}
