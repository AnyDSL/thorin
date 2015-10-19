#ifndef PLATFORM_H
#define PLATFORM_H

#include <unordered_map>
#include <mutex>
#include <string>
#include <cassert>

#include "thorin_runtime.h"

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
    /// Releases memory for a device on this platform.
    virtual void release(device_id dev, void* ptr, int64_t size) = 0;

    /// Maps a region of memory, with the given offset and size in bytes.
    virtual void* map(device_id dev, void* ptr, int64_t offset, int64_t size) = 0;
    /// Unmaps a region of memory. The pointer to the origin of the buffer is also provided.
    virtual void unmap(device_id dev, void* view, void* ptr) = 0;

    /// Sets the kernel launch block size.
    virtual void set_block_size(device_id dev, int32_t x, int32_t y, int32_t z) = 0;
    /// Sets the kernel launch grid size.
    virtual void set_grid_size(device_id dev, int32_t x, int32_t y, int32_t z) = 0;
    /// Sets the argument of a kernel.
    virtual void set_kernel_arg(device_id dev, int32_t arg, void* ptr, int32_t size) = 0;
    /// Loads a kernel on a device (taken from a file).
    virtual void load_kernel(device_id dev, const char* file, const char* name) = 0;
    /// Launches the loaded kernel.
    virtual void launch_kernel(device_id dev) = 0;
    /// Waits for the completion of all the launched kernels on the given device.
    virtual void synchronize(device_id dev) = 0;

    /// Copies memory. Copy can only be performed devices in the same platform.
    virtual void copy(const void* src, void* dst) = 0;
    /// Copies memory from the host (CPU).
    virtual void copy_from_host(const void* src, void* dst) = 0;
    /// Copies memory to the host (CPU).
    virtual void copy_to_host(const void* src, void* dst) = 0;

    /// Returns the number of devices in this platform.
    virtual int dev_count() = 0;
    /// Returns the platform name.
    virtual std::string name() = 0;

protected:
    Runtime* runtime_;
};

#endif
