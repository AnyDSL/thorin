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
bool World::load_plugin(const char* plugin_name) {
#ifdef _WIN32
    return false;
#else
    void *handle = dlopen(plugin_name, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        ELOG("Error loading plugin {}: {}", plugin_name, dlerror());
        ELOG("Is plugin contained in LD_LIBRARY_PATH?");
        return false;
    }
    dlerror();

    char *error;
    auto initfunc = reinterpret_cast<plugin_init_func_t*>(dlsym(handle, "init"));
    if ((error = dlerror()) != NULL) {
        ILOG("Plugin {} did not supply an init function", plugin_name);
    } else {
        initfunc();
    }

    data_.plugin_modules_.push_back(handle);
    return true;
#endif
}

template <typename T>
static T* lookup_plugin_function(void* plugin_module, const char* function_name) {
#ifdef _WIN32
#else
    if (void* func = dlsym(plugin_module, function_name)) {
        return reinterpret_cast<T*>(func);
    }
#endif
    return nullptr;
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
