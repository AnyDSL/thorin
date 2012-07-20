#ifndef ANYDSL_ARRAYREF_H
#define ANYDSL_ARRAYREF_H

#include <iterator>

namespace anydsl {

template<class LEFT, class RIGHT>
LEFT& deref_hook(RIGHT* ptr) {
    return *ptr;
}

template<class T, class Deref = T, Deref& (*Hook)(T*) = deref_hook<T, T> >
class ArrayRef {
public:

    class const_iterator {
    public:

        typedef std::random_access_iterator_tag iterator_category;
        typedef const Deref value_type;
        typedef ptrdiff_t difference_type;
        typedef const T* pointer;
        typedef const Deref& reference;

        const_iterator(const const_iterator& i) : base_(i.base_) {}
        const_iterator(T* base) : base_(base) {}

        const_iterator& operator ++ () { ++base_; return *this; }
        const_iterator  operator ++ (int) { const_iterator i(*this); ++(*this); return i; }

        const_iterator& operator -- () { --base_; return *this; }
        const_iterator  operator -- (int) { const_iterator i(*this); --(*this); return i; }

        const_iterator operator + (difference_type d) { return const_iterator(base_ + d); }
        const_iterator operator - (difference_type d) { return const_iterator(base_ - d); }

        bool operator <  (const const_iterator& i) { return base_ <  i.base_; }
        bool operator <= (const const_iterator& i) { return base_ <= i.base_; }
        bool operator >  (const const_iterator& i) { return base_ >  i.base_; }
        bool operator >= (const const_iterator& i) { return base_ >= i.base_; }
        bool operator == (const const_iterator& i) { return base_ == i.base_; }
        bool operator != (const const_iterator& i) { return base_ != i.base_; }

        const Deref& operator *  () { return Hook(base_); }
        const Deref& operator -> () { return Hook(base_); }

        T* base() const { return base_; }

    private:

        T* base_;
    };

    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    ArrayRef(T* ptr, size_t size)
        : ptr_(ptr) 
        , size_(size)
    {}

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    T const& operator [] (size_t i) const {
        anydsl_assert(i < size(), "index out of bounds");
        return ptr_[i];
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T& front() const { assert(!empty()); return ptr_[0]; }
    T& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

private:

    T* ptr_;
    size_t size_;
};

} // namespace anydsl

#endif
