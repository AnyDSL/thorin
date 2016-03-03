#ifndef THORIN_AUTOPTR_HEADER
#define THORIN_AUTOPTR_HEADER

#include <vector>

namespace thorin {

/// Similar to \c std::unique_ptr<T> but some more convenience like cast operators and non-explicit constructors.
template<class T>
class AutoPtr {
public:
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
    AutoPtr(const AutoPtr<T>& aptr); // forbid copy constructor
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
inline T& lazy_init(const This* self, AutoPtr<T>& ptr) {
    return *(ptr ? ptr : ptr = new T(*self));
}

}

#endif
