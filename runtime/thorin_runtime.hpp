#ifndef THORIN_RUNTIME_HPP
#define THORIN_RUNTIME_HPP

#ifndef THORIN_RUNTIME_H
#include "thorin_runtime.h"
#endif

namespace thorin {

enum class Platform : int32_t {
    HOST = THORIN_HOST,
    CUDA = THORIN_CUDA,
    OPENCL = THORIN_OPENCL
};

struct Device {
    Device(int32_t id) : id(id) {}
    int32_t id;
};

template <typename T>
class Array {
    template <typename U, typename V> friend void copy(const Array<U>&, Array<V>&);
public:
    Array()
        : platform_(Platform::HOST),
          device_(Device(0)),
          size_(0), data_(nullptr)
    {}

    Array(int64_t size)
        : Array(Platform::HOST, Device(0), size)
    {}

    Array(Platform p, Device d, T* ptr)
        : platform_(p), device_(d), data_(ptr)
    {}

    Array(Platform p, Device d, int64_t size)
        : platform_(p), device_(d) {
        allocate(p, d, size);
    }

    Array(Array&& other)
        : platform_(other.platform_),
          device_(other.device_),
          size_(other.size_),
          data_(other.data_) {
        other.data_ = nullptr;
    }

    Array& operator = (Array&& other) {
        deallocate();
        platform_ = other.platform_;
        device_ = other.device_;
        size_ = other.size_;
        data_ = other.data_;
        other.data_ = nullptr;
        return *this;
    }

    Array(const Array&) = delete;
    Array& operator = (const Array&) = delete;

    ~Array() { deallocate(); }

    T* begin() { return data_; }
    const T* begin() const { return data_; }
    
    T* end() { return data_ + size_; }
    const T* end() const { return data_ + size_; }

    T* data() { return data_; }
    const T* data() const { return data_; }

    Platform platform() const { return platform_; }
    Device device() const { return device_; }
    int64_t size() const { return size_; }

    const T& operator [] (int i) const { return data_[i]; }
    T& operator [] (int i) { return data_[i]; }

    T* release() {
        T* ptr = data_;
        data_ = nullptr;
        size_ = 0;
        platform_ = Platform::HOST;
        device_ = Device(0);
        return ptr;
    }

protected:
    void allocate(Platform p, Device d, int64_t size) {
        size_ = size;
        data_ = (T*)thorin_alloc((int32_t)p, d.id, sizeof(T) * size);
    }

    void deallocate() {
        if (data_) thorin_release((void*)data_);
    }

    T* data_;
    int64_t size_;
    Platform platform_;
    Device device_;
};

template <typename T, typename U>
void copy(const Array<T>& a, Array<U>& b) {
    thorin_copy((const void*)a.data_, (void*)b.data_);
}

} // namespace thorin

#endif
