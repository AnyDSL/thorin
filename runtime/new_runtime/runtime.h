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

    struct View {
        void* ptr;
        int64_t off;
        int64_t size;

        View() {}
        View(void* ptr, int64_t off, int64_t size)
            : ptr(ptr), off(off), size(size)
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
        void* ptr = platforms_[plat]->alloc(dev, size);
        mems_.emplace(ptr, Mem(plat, dev, size));
        return ptr;
    }

    /// Releases memory.
    void release(void* ptr) {
        auto it = mems_.find(ptr);
        if (it == mems_.end())
            error("Memory not allocated from runtime");

        platforms_[it->second.plat]->release(it->second.dev, ptr, it->second.size);
        mems_.erase(it);
    }

    void set_block_size(platform_id plat, device_id dev, uint32_t x, uint32_t y, uint32_t z) {
        platforms_[plat]->set_block_size(dev, x, y, z);
    }

    void set_grid_size(platform_id plat, device_id dev, uint32_t x, uint32_t y, uint32_t z) {
        platforms_[plat]->set_grid_size(dev, x, y, z); 
    }

    void set_arg(platform_id plat, device_id dev, uint32_t arg, void* ptr) {
        platforms_[plat]->set_arg(dev, arg, ptr);
    }

    void load_kernel(platform_id plat, device_id dev, const char* file, const char* name) {
        platforms_[plat]->load_kernel(dev, file, name);
    }

    void launch_kernel(platform_id plat, device_id dev) {
        platforms_[plat]->launch_kernel(dev);
    }

    /// Copies memory.
    void copy(const void* src, void* dst) {
        auto src_mem = mems_.find((void*)src);
        auto dst_mem = mems_.find(dst);
        if (src_mem == mems_.end() || dst_mem == mems_.end())
            error("Memory not allocated from runtime");

        if (src_mem->second.plat == dst_mem->second.plat) {
            // Copy from same platform
            platforms_[src_mem->second.plat]->copy(src, dst);
        } else {
            // Copy from another platform
            if (src_mem->second.plat == 0) {
                // Source is the CPU platform
                platforms_[dst_mem->second.plat]->copy_from_host(src, dst);
            } else if (dst_mem->second.dev == 0) {
                // Destination is the CPU platform
                platforms_[src_mem->second.plat]->copy_to_host(src, dst);
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
    std::unordered_map<void*, Mem> mems_;
};

#endif
