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
    void release(device_id, void*, int64_t) override { platform_error(); }
    void* map(void*, int64_t, int64_t) override { platform_error(); return nullptr; }
    void unmap(void*) override { platform_error(); }

    void set_block_size(device_id, uint32_t, uint32_t, uint32_t) { platform_error(); }
    void set_grid_size(device_id, uint32_t, uint32_t, uint32_t) { platform_error(); }
    void set_arg(device_id, uint32_t, void*, uint32_t) override { platform_error(); }
    void load_kernel(device_id, const char*, const char*) { platform_error(); }
    void launch_kernel(device_id) override { platform_error(); }

    void copy(const void* src, void* dst) override { platform_error(); }
    void copy_from_host(const void* src, void* dst) override { platform_error(); }
    void copy_to_host(const void* src, void* dst) override { platform_error(); }

    int dev_count() override { return 0; }

    std::string name() override { return name_; }

    std::string name_;
};

#endif
