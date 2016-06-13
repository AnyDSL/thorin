#ifndef PLATFORM_H
#define PLATFORM_H

#include <cstdint>
#include <string>

#include "thorin/util/log.h"

class Runtime;
enum device_id : unsigned {};
enum platform_id : unsigned {};

/// A runtime platform. Exposes a set of devices, a copy function,
/// and functions to allocate and release memory.
class Platform {
public:
    Platform(Runtime* runtime)
        : runtime_(runtime)
    {}

    virtual ~Platform() {}

    /// Allocates memory for a device on this platform.
    virtual void* alloc(device_id dev, int64_t size) = 0;
    /// Allocates page-locked host memory for a platform (and a device).
    virtual void* alloc_host(device_id dev, int64_t size) = 0;
    /// Allocates unified memory for a platform (and a device).
    virtual void* alloc_unified(device_id dev, int64_t size) = 0;
    /// Returns the device memory associated with the page-locked memory.
    virtual void* get_device_ptr(device_id dev, void* ptr) = 0;
    /// Releases memory for a device on this platform.
    virtual void release(device_id dev, void* ptr) = 0;
    /// Releases page-locked host memory for a device on this platform.
    virtual void release_host(device_id dev, void* ptr) = 0;

    /// Sets the kernel launch block size.
    virtual void set_block_size(device_id dev, int32_t x, int32_t y, int32_t z) = 0;
    /// Sets the kernel launch grid size.
    virtual void set_grid_size(device_id dev, int32_t x, int32_t y, int32_t z) = 0;
    /// Sets the argument of a kernel. The argument is a pointer to the value on the stack.
    virtual void set_kernel_arg(device_id dev, int32_t arg, void* ptr, int32_t size) = 0;
    /// Sets the argument of a kernel. The argument is a pointer to device-allocated memory.
    virtual void set_kernel_arg_ptr(device_id dev, int32_t arg, void* ptr) = 0;
    /// Sets the argument of a kernel. The argument is a pointer to a stack-allocated structure.
    virtual void set_kernel_arg_struct(device_id dev, int32_t arg, void* ptr, int32_t size) = 0;
    /// Loads a kernel on a device (taken from a file).
    virtual void load_kernel(device_id dev, const char* file, const char* name) = 0;
    /// Launches the loaded kernel.
    virtual void launch_kernel(device_id dev) = 0;
    /// Waits for the completion of all the launched kernels on the given device.
    virtual void synchronize(device_id dev) = 0;

    /// Copies memory. Copy can only be performed devices in the same platform.
    virtual void copy(device_id dev_src, const void* src, int64_t offset_src, device_id dev_dst, void* dst, int64_t offset_dst, int64_t size) = 0;
    /// Copies memory from the host (CPU).
    virtual void copy_from_host(const void* src, int64_t offset_src, device_id dev_dst, void* dst, int64_t offset_dst, int64_t size) = 0;
    /// Copies memory to the host (CPU).
    virtual void copy_to_host(device_id dev_src, const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size) = 0;

    /// Returns the number of devices in this platform.
    virtual int dev_count() = 0;
    /// Returns the platform name.
    virtual std::string name() = 0;

protected:
    void platform_error() {
        ELOG("The selected platform is not available");
    }

    Runtime* runtime_;
};

#endif
