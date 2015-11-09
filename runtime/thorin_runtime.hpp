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

inline int32_t make_device(Platform p, Device d) {
    return THORIN_DEVICE((int32_t)p, d.id);
}

template <typename T>
class Array {
    template <typename U> friend void copy(const Array<U>&, Array<U>&);
    template <typename U> friend void copy(const Array<U>&, Array<U>&, int64_t);
    template <typename U> friend void copy(const Array<U>&, int64_t, Array<U>&, int64_t, int64_t);

public:
    Array()
        : dev_(0), size_(0), data_(nullptr)
    {}

    Array(int64_t size)
        : Array(Platform::HOST, Device(0), size)
    {}

    Array(int32_t dev, T* ptr, int64_t size)
        : dev_(dev), data_(ptr), size_(size)
    {}

    Array(Platform p, Device d, int64_t size)
        : dev_(make_device(p, d)) {
        allocate(size);
    }

    Array(Array&& other)
        : dev_(other.dev_),
          size_(other.size_),
          data_(other.data_) {
        other.data_ = nullptr;
    }

    Array& operator = (Array&& other) {
        deallocate();
        dev_ = other.dev_;
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

    int64_t size() const { return size_; }

    const T& operator [] (int i) const { return data_[i]; }
    T& operator [] (int i) { return data_[i]; }

    T* release() {
        T* ptr = data_;
        data_ = nullptr;
        size_ = 0;
        dev_ = 0;
        return ptr;
    }

protected:
    void allocate(int64_t size) {
        size_ = size;
        data_ = (T*)thorin_alloc(dev_, sizeof(T) * size);
    }

    void deallocate() {
        if (data_) thorin_release(dev_, (void*)data_);
    }

    T* data_;
    int64_t size_;
    int32_t dev_;
};

template <typename T>
void copy(const Array<T>& a, Array<T>& b) {
    thorin_copy(a.dev_, (const void*)a.data_, 0,
                b.dev_, (void*)b.data_, 0,
                a.size_ * sizeof(T));
}

template <typename T>
void copy(const Array<T>& a, Array<T>& b, int64_t size) {
    thorin_copy(a.dev_, (const void*)a.data_, 0,
                b.dev_, (void*)b.data_, 0,
                size * sizeof(T));
}

template <typename T>
void copy(const Array<T>& a, int64_t offset_a, Array<T>& b, int64_t offset_b, int64_t size) {
    thorin_copy(a.dev_, (const void*)a.data_, offset_a,
                b.dev_, (void*)b.data_, offset_b,
                size * sizeof(T));
}

} // namespace thorin

#endif
