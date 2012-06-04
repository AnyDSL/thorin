#ifndef DSLU_AUTOPTR_HEADER
#define DSLU_AUTOPTR_HEADER

namespace anydsl {

template<class T>
class AutoPtr {
public:

    explicit AutoPtr(T* ptr = 0)
        : ptr_(ptr)
    {}
    ~AutoPtr() { delete ptr_; }

    void release() {
        T* ptr = ptr_;
        ptr_ = 0;
        return ptr;
    }

    operator T*() const { return ptr_; }
    T* operator -> () const { return ptr_; }

    AutoPtr<T>& operator = (T* ptr) {
        delete ptr_;
        ptr_ = ptr;
        return *this;
    }

private:

    // forbid copy constructor and standard assignment operator
    AutoPtr(const AutoPtr<T>&);
    AutoPtr<T>& operator = (const AutoPtr<T>&);

    T* ptr_;
};


}

#endif

