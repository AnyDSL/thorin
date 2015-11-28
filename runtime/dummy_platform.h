#ifndef DUMMY_PLATFORM_H
#define DUMMY_PLATFORM_H

#include "platform.h"
#include "runtime.h"

/// Dummy platform, implemented
class DummyPlatform : public Platform {
public:
    DummyPlatform(Runtime* runtime, const std::string& name)
        : Platform(runtime), name_(name)
    {}

protected:
    void platform_error() {
        runtime_->error("The selected platform is not available");
    }

    void* alloc(device_id, int64_t) override { platform_error(); return nullptr; }
    void* alloc_unified(device_id, int64_t) override { platform_error(); return nullptr; }
    void release(device_id, void*) override { platform_error(); }

    void set_block_size(device_id, int32_t, int32_t, int32_t) override { platform_error(); }
    void set_grid_size(device_id, int32_t, int32_t, int32_t) override { platform_error(); }
    void set_kernel_arg(device_id, int32_t, void*, int32_t) override { platform_error(); }
    void set_kernel_arg_ptr(device_id, int32_t, void*) override { platform_error(); }
    void set_kernel_arg_struct(device_id, int32_t, void*, int32_t) override { platform_error(); }
    void load_kernel(device_id, const char*, const char*) override { platform_error(); }
    void launch_kernel(device_id) override { platform_error(); }
    void synchronize(device_id) override { platform_error(); }

    void copy(device_id, const void*, int64_t, device_id, void*, int64_t, int64_t) override { platform_error(); }
    void copy_from_host(const void*, int64_t, device_id, void*, int64_t, int64_t) override { platform_error(); }
    void copy_to_host(device_id, const void*, int64_t, void*, int64_t, int64_t) override { platform_error(); }

    int dev_count() override { return 0; }

    std::string name() override { return name_; }

    std::string name_;
};

#endif
