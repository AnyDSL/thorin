#ifndef RUNTIME_H
#define RUNTIME_H

#include <iostream>

#include "platform.h"

class Runtime {
public:
    struct Mem {
        platform_id plat;
        device_id dev;
        int64_t size;
        Mem() {}
        Mem(platform_id plat, device_id dev, int64_t size)
            : plat(plat), dev(dev), size(size)
        {}
    };

    Runtime();

    ~Runtime() {
        assert(mems_.size() == 0 && "Some memory blocks have not been released");
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
        void* ptr = platforms_[plat]->alloc(dev, size);
        mems_.emplace(ptr, Mem(plat, dev, size));
        return ptr;
    }

    /// Releases memory.
    void release(void* ptr) {
        auto it = mems_.find(ptr);
        if (it == mems_.end())
            error("Memory not allocated by the runtime");

        platforms_[it->second.plat]->release(it->second.dev, ptr, it->second.size);
        mems_.erase(it);
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
    void copy(const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size) {
        auto src_mem = mems_.find((void*)src);
        auto dst_mem = mems_.find(dst);
        if (src_mem == mems_.end() || dst_mem == mems_.end())
            error("Memory not allocated from runtime");

        if (src_mem->second.plat == dst_mem->second.plat) {
            // Copy from same platform
            platforms_[src_mem->second.plat]->copy(src, offset_src, dst, offset_dst, size);
        } else {
            // Copy from another platform
            if (src_mem->second.plat == 0) {
                // Source is the CPU platform
                platforms_[dst_mem->second.plat]->copy_from_host(src, offset_src, dst, offset_dst, size);
            } else if (dst_mem->second.dev == 0) {
                // Destination is the CPU platform
                platforms_[src_mem->second.plat]->copy_to_host(src, offset_src, dst, offset_dst, size);
            } else {
                error("Cannot copy memory between different platforms");
            }
        }
    }

    Mem memory_info(const void* ptr) {
        auto it = mems_.find((void*)ptr);
        assert(it != mems_.end());
        return it->second;
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
    std::unordered_map<void*, Mem>  mems_;
};

#endif
