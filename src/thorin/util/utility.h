#ifndef THORIN_UTILITY_H
#define THORIN_UTILITY_H

#include <memory>
#include <queue>
#include <vector>

namespace thorin {

/// Similar to \c std::unique_ptr<T> but some more convenience like cast operators and non-explicit constructors.
template<class T>
class AutoPtr {
public:
    AutoPtr(const AutoPtr<T>&) = delete;

    AutoPtr(T* ptr = nullptr)
        : ptr_(ptr)
    {}
    AutoPtr(AutoPtr<T>&& aptr)
        : ptr_(std::move(aptr.ptr_))
    {
        aptr.ptr_ = nullptr; // take ownership
    }
    ~AutoPtr() { delete ptr_; }

    void release() { delete ptr_; ptr_ = nullptr; }
    T* get() const { return ptr_; }
    T** address() { return &ptr_; }
    T* const* address() const { return &ptr_; }
    operator bool() const { return ptr_ != nullptr; }
    operator T*() const {return ptr_; }
    T* operator->() const { return ptr_; }
    AutoPtr<T>& operator=(AutoPtr<T> other) { swap(*this, other); return *this; }
    friend void swap(AutoPtr<T>& a, AutoPtr<T>& b) { using std::swap; swap(a.ptr_, b.ptr_); }

private:
    T* ptr_;
};

/// A simple wrapper around a usual pointer but initialized with nullptr and checked via assert if valid prior to dereferencing.
template<class T>
class SafePtr {
public:
    SafePtr(const SafePtr<T>& sptr)
        : ptr_(sptr.ptr_)
    {}
    SafePtr(SafePtr<T>&& sptr)
        : ptr_(std::move(sptr.ptr_))
    {}
    SafePtr(T* ptr = nullptr)
        : ptr_(ptr)
    {}

    T* get() const { assert(ptr_ != nullptr); return ptr_; }
    T** address() { return &ptr_; }
    T* const* address() const { return &ptr_; }
    operator bool() const { return ptr_ != nullptr; }
    operator T*() const {return get(); }
    T* operator->() const { return get(); }
    SafePtr<T>& operator=(SafePtr<T> other) { swap(*this, other); return *this; }
    friend void swap(SafePtr<T>& a, SafePtr<T>& b) { using std::swap; swap(a.ptr_, b.ptr_); }

private:
    T* ptr_;
};

/// Like \c std::vector<T*> but \p AutoVector deletes its managed objects.
template<class T>
class AutoVector : public std::vector<T> {
public:
    AutoVector()
        : std::vector<T>()
    {}
    explicit AutoVector(typename std::vector<T>::size_type s)
        : std::vector<T>(s)
    {}
    ~AutoVector() { for (auto p : *this) delete p; }
};

/// Use to initialize an \p AutoPtr in a lazy way.
template<class This, class T>
inline T& lazy_init(const This* self, std::unique_ptr<T>& ptr) {
    return *(ptr ? ptr : (ptr.reset(new T(*self)), ptr));
}

template<class T>
T pop(std::queue<T>& queue) {
    auto val = queue.front();
    queue.pop();
    return val;
}

template<class T>
struct Push {
    Push(T& t, T new_val)
        : old_(t)
        , ref_(t)
    {
        t = new_val;
    }
    ~Push() { ref_ = old_; }

private:
    T old_;
    T& ref_;
};

template<class T, class U>
inline Push<T> push(T& t, U new_val) { return Push<T>(t, new_val); }

#define THORIN_LNAME__(name, line) name##__##line
#define THORIN_LNAME_(name, line)  THORIN_LNAME__(name, line)
#define THORIN_LNAME(name)         THORIN_LNAME_(name, __LINE__)

#define THORIN_PUSH(what, with) auto THORIN_LNAME(thorin_push) = thorin::push((what), (with))

/// Determines whether @p i is a power of two.
constexpr size_t is_power_of_2(size_t i) { return ((i != 0) && !(i & (i - 1))); }

constexpr unsigned log2(unsigned n, unsigned p = 0) { return (n <= 1) ? p : log2(n / 2, p + 1); }

inline uint32_t round_to_power_of_2(uint32_t i) {
    i--;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i++;
    return i;
}

}

#endif
