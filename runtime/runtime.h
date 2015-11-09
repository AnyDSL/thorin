#ifndef RUNTIME_H
#define RUNTIME_H

#include <iostream>

#include "platform.h"

class Runtime {
public:
    Runtime();

    ~Runtime() {
        for (auto p: platforms_) {
            delete p;
        }
    }

    /// Registers the given platform into the runtime.
    template <typename T, typename... Args>
    void register_platform(Args... args) {
        Platform* p = new T(this, args...);
        platforms_.push_back(p);
    }

    /// Displays available platforms.
    void display_info(std::ostream& os) {
        os << "Available platforms:" << std::endl;
        for (auto p: platforms_) {
            os << "    * " << p->name() << ": " << p->dev_count() << " device(s)" << std::endl;
        }
    }

    /// Allocates memory on the given device.
    void* alloc(platform_id plat, device_id dev, int64_t size) {
        check_device(plat, dev);
        return platforms_[plat]->alloc(dev, size);
    }

    /// Releases memory.
    void release(platform_id plat, device_id dev, void* ptr) {
        platforms_[plat]->release(dev, ptr);
    }

    void set_block_size(platform_id plat, device_id dev, int32_t x, int32_t y, int32_t z) {
        check_device(plat, dev);
        platforms_[plat]->set_block_size(dev, x, y, z);
    }

    void set_grid_size(platform_id plat, device_id dev, int32_t x, int32_t y, int32_t z) {
        check_device(plat, dev);
        platforms_[plat]->set_grid_size(dev, x, y, z); 
    }

    void set_kernel_arg(platform_id plat, device_id dev, int32_t arg, void* ptr, int32_t size) {
        check_device(plat, dev);
        platforms_[plat]->set_kernel_arg(dev, arg, ptr, size);
    }

    void set_kernel_arg_ptr(platform_id plat, device_id dev, int32_t arg, void* ptr) {
        check_device(plat, dev);
        platforms_[plat]->set_kernel_arg_ptr(dev, arg, ptr);
    }

    void set_kernel_arg_struct(platform_id plat, device_id dev, int32_t arg, void* ptr, int32_t size) {
        check_device(plat, dev);
        platforms_[plat]->set_kernel_arg_struct(dev, arg, ptr, size);
    }

    void load_kernel(platform_id plat, device_id dev, const char* file, const char* name) {
        check_device(plat, dev);
        platforms_[plat]->load_kernel(dev, file, name);
    }

    void launch_kernel(platform_id plat, device_id dev) {
        check_device(plat, dev);
        platforms_[plat]->launch_kernel(dev);
    }

    void synchronize(platform_id plat, device_id dev) {
        check_device(plat, dev);
        platforms_[plat]->synchronize(dev);
    }

    /// Copies memory.
    void copy(platform_id plat_src, device_id dev_src, const void* src, int64_t offset_src,
              platform_id plat_dst, device_id dev_dst, void* dst, int64_t offset_dst, int64_t size) {
        if (plat_src == plat_dst) {
            // Copy from same platform
            platforms_[plat_src]->copy(dev_src, src, offset_src, dev_dst, dst, offset_dst, size);
        } else {
            // Copy from another platform
            if (plat_src == 0) {
                // Source is the CPU platform
                platforms_[plat_dst]->copy_from_host(src, offset_src, dev_dst, dst, offset_dst, size);
            } else if (plat_dst == 0) {
                // Destination is the CPU platform
                platforms_[plat_src]->copy_to_host(dev_src, src, offset_src, dst, offset_dst, size);
            } else {
                error("Cannot copy memory between different platforms");
            }
        }
    }

    template <typename... Args>
    void error(Args... args) {
        std::cerr << "Runtime error: ";
        print(args...);
        std::abort();
    }

    template <typename... Args>
    void log(Args... args) {
#ifndef NDEBUG
        std::cerr << "Runtime message: ";
        print(args...);
#endif
    }

private:
    void check_device(platform_id plat, device_id dev) {
        assert((int)dev < platforms_[plat]->dev_count() && "Invalid device");
    }

    template <typename T, typename... Args>
    static void print(T t, Args... args) {
        std::cerr << t;
        print(args...);
    }

    template <typename T>
    static void print(T t) {
        std::cerr << t << std::endl;
    }

    std::vector<Platform*> platforms_;
};

#endif
