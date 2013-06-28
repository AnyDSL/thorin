#ifndef ANYDSL2_AUTOPTR_HEADER
#define ANYDSL2_AUTOPTR_HEADER

#include <vector>

namespace anydsl2 {

template<class T>
class AutoPtr {
public:

    AutoPtr(T* ptr = 0)
        : ptr_(ptr)
    {}
    ~AutoPtr() { delete ptr_; }
    AutoPtr(const AutoPtr<T>& aptr)
        : ptr_(aptr.get())
    {
        const_cast<AutoPtr<T>&>(aptr).ptr_ = 0; // take ownership
    }


    void release() { delete ptr_; ptr_ = 0; }
    T* get() const { return ptr_; }
    operator T*() const { return ptr_; }
    T* operator -> () const { return ptr_; }
    AutoPtr<T>& operator = (T* ptr) { delete ptr_; ptr_ = ptr; return *this; }

private:

    // forbid copy constructor and standard assignment operator
    AutoPtr<T>& operator = (const AutoPtr<T>&);

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
