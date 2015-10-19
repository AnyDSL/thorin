#ifndef CPU_PLATFORM_H
#define CPU_PLATFORM_H

#include "platform.h"
#include "runtime.h"

#include <cstring>

/// CPU platform, allocation is guaranteed to be aligned to 64 bytes.
class CpuPlatform : public Platform {
public:
    CpuPlatform(Runtime* runtime)
        : Platform(runtime)
    {}

protected:
    void* alloc(device_id dev, int64_t size) override {
        return thorin_aligned_malloc(size, 64);
    }

    void release(device_id, void* ptr, int64_t) override {
        thorin_aligned_free(ptr);
    }

    void* map(void* ptr, int64_t offset, int64_t) override {
        return (void*)((int8_t*)ptr + offset);
    }

    void unmap(void* view) override {}

    void no_kernel() { runtime_->error("Kernels are not supported on the CPU"); }

    void set_block_size(device_id, int32_t, int32_t, int32_t) override { no_kernel(); }
    void set_grid_size(device_id, int32_t, int32_t, int32_t) override { no_kernel(); }
    void set_kernel_arg(device_id, int32_t, void*, int32_t) override { no_kernel(); }
    void load_kernel(device_id, const char*, const char*) override { no_kernel(); }
    void launch_kernel(device_id) override { no_kernel(); }
    void synchronize(device_id dev) override { no_kernel(); }

    void copy(const void* src, void* dst) override {
        auto info = runtime_->memory_info(src);
        memcpy(dst, src, info.size);
    }
    void copy_from_host(const void* src, void* dst) override { copy(src, dst); }
    void copy_to_host(const void* src, void* dst) override { copy(src, dst); }

    int dev_count() override { return 1; }

    std::string name() override { return "CPU"; }
};

#endif
