#ifndef ANYDSL_ARRAY_H
#define ANYDSL_ARRAY_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include <vector>

#include "anydsl/util/assert.h"

namespace anydsl {

template<class T> class Array;

template<class LEFT, class RIGHT>
inline LEFT& deref_hook(RIGHT* ptr) {
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

        difference_type operator + (const_iterator i) { return difference_type(base_ + i.base()); }
        difference_type operator - (const_iterator i) { return difference_type(base_ - i.base()); }

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

    template<size_t N>
    ArrayRef(T (&array)[N]) 
        : ptr_(&array[0])
        , size_(N)
    {}
    ArrayRef(std::vector<T>& vector)
        : ptr_(&*vector.begin())
        , size_(vector.size())
    {}
    ArrayRef(const std::vector<T>& vector)
        : ptr_(&*vector.begin())
        , size_(vector.size())
    {}
    ArrayRef(T* ptr, size_t size)
        : ptr_(ptr) 
        , size_(size)
    {}

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    Deref const& operator [] (size_t i) const {
        anydsl_assert(i < size(), "index out of bounds");
        return Hook(ptr_ + i);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T& front() const { assert(!empty()); return ptr_[0]; }
    T& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_front(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_back(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }

    operator ArrayRef<T> () { return ArrayRef<T>(ptr_, size_); }

private:

    T* ptr_;
    size_t size_;
};


template<class T>
class Array {
public:

    explicit Array(size_t size)
        : ptr_(new T[size])
        , size_(size)
    {}
    explicit Array(ArrayRef<T> ref)
        : ptr_(new T[ref.size()])
        , size_(ref.size())
    {
        std::memcpy(ptr_, ref.begin().base(), size() * sizeof(T));
    }
    Array(const Array<T>& array) 
        : ptr_(new T[array.size()])
        , size_(array.size())
    {
        std::memcpy(ptr_, array.ptr_, size() * sizeof(T));
    }

    ~Array() { delete[] ptr_; }


    typedef T* iterator;
    typedef const T* const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    iterator begin() { return ptr_; }
    iterator end() { return ptr_ + size_; }
    reverse_iterator rbegin() { return const_reverse_iterator(end()); }
    reverse_iterator rend() { return const_reverse_iterator(begin()); }
    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    T& operator [] (size_t i) { anydsl_assert(i < size(), "index out of bounds"); return ptr_[i]; }
    T const& operator [] (size_t i) const { anydsl_assert(i < size(), "index out of bounds"); return ptr_[i]; }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T& front() const { assert(!empty()); return ptr_[0]; }
    T& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_front(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_back(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }

    operator ArrayRef<T>() { return ArrayRef<T>(ptr_, size_); }

private:

    Array<T>& operator = (const Array<T>& array);

    T* ptr_;
    size_t size_;
};

} // namespace anydsl

#endif
