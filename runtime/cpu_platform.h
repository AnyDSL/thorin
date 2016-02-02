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
    void* alloc(device_id, int64_t size) override {
        return thorin_aligned_malloc(size, 64);
    }

    void* alloc_host(device_id dev, int64_t size) override {
        return alloc(dev, size);
    }

    void* alloc_unified(device_id dev, int64_t size) override {
        return alloc(dev, size);
    }

    void* get_device_ptr(device_id, void* ptr) override {
        return ptr;
    }

    void release(device_id, void* ptr) override {
        thorin_aligned_free(ptr);
    }

    void release_host(device_id dev, void* ptr) override {
        release(dev, ptr);
    }

    void no_kernel() { ELOG("Kernels are not supported on the CPU"); }

    void set_block_size(device_id, int32_t, int32_t, int32_t) override { no_kernel(); }
    void set_grid_size(device_id, int32_t, int32_t, int32_t) override { no_kernel(); }
    void set_kernel_arg(device_id, int32_t, void*, int32_t) override { no_kernel(); }
    void set_kernel_arg_ptr(device_id, int32_t, void*) override { no_kernel(); }
    void set_kernel_arg_struct(device_id, int32_t, void*, int32_t) override { no_kernel(); }
    void load_kernel(device_id, const char*, const char*) override { no_kernel(); }
    void launch_kernel(device_id) override { no_kernel(); }
    void synchronize(device_id) override { no_kernel(); }

    void copy(const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size) {
        memcpy((char*)dst + offset_dst, (char*)src + offset_src, size);
    }

    void copy(device_id, const void* src, int64_t offset_src,
              device_id, void* dst, int64_t offset_dst, int64_t size) override {
        copy(src, offset_src, dst, offset_dst, size);
    }
    void copy_from_host(const void* src, int64_t offset_src, device_id,
                        void* dst, int64_t offset_dst, int64_t size) override {
        copy(src, offset_src, dst, offset_dst, size);
    }
    void copy_to_host(device_id, const void* src, int64_t offset_src,
                      void* dst, int64_t offset_dst, int64_t size) override {
        copy(src, offset_src, dst, offset_dst, size);
    }

    int dev_count() override { return 1; }

    std::string name() override { return "CPU"; }
};

#endif
