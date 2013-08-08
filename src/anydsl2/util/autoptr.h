#ifndef ANYDSL2_AUTOPTR_HEADER
#define ANYDSL2_AUTOPTR_HEADER

#include <vector>

namespace anydsl2 {

/// Similar to std::unique_ptr<T> but some more convenience like cast operators and non-explicit constructors.
template<class T>
class AutoPtr {
public:

    AutoPtr(T* ptr = 0)
        : ptr_(ptr)
    {}
    ~AutoPtr() { delete ptr_; }
    AutoPtr(AutoPtr<T>&& aptr)
        : ptr_(aptr.get())
    {
        aptr.ptr_ = 0; // take ownership
    }

    void release() { delete ptr_; ptr_ = 0; }
    T* get() const { return ptr_; }
    operator T*() const { return ptr_; }
    T* operator -> () const { return ptr_; }
    AutoPtr<T>& operator = (T* ptr) { delete ptr_; ptr_ = ptr; return *this; }
    AutoPtr<T>& operator = (AutoPtr<T> other) { swap(*this, other); return *this; }
    friend void swap(AutoPtr<T>& a, AutoPtr<T>& b) { std::swap(a.ptr_, b.ptr_); }

private:

    AutoPtr(const AutoPtr<T>& aptr); // forbid copy constructor

    T* ptr_;
};

template<class T>
class AutoVector : public std::vector<T> {
public:

    AutoVector() : std::vector<T>() {}

    explicit AutoVector(typename std::vector<T>::size_type s)
        : std::vector<T>(s)
    {}

    ~AutoVector() {
        for (typename std::vector<T>::const_iterator i = this->begin(), e = this->end(); i != e; ++i)
            delete *i;
    }
};

}

#endif
