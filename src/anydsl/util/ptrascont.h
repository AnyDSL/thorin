#ifndef ANYDSL_PTRASCONT_H
#define ANYDSL_PTRASCONT_H

namespace anydsl {

template<class T>
class PtrAsCont {
public:
    typedef T** const_iterator;
    typedef std::reverse_iterator<T**> const_reverse_iterator;

    PtrAsCont(T** ptr, size_t size)
        : ptr_(ptr) 
        , size_(size)
    {}

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    T* const& operator [] (size_t i) const {
        anydsl_assert(i < size(), "index out of bounds");
        return ptr_[i];
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T*& front() const { assert(!empty()); return ptr_[0]; }
    T*& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

private:

    T** ptr_;
    size_t size_;
};

} // namespace anydsl

#endif
