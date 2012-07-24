#ifndef ANYDSL_ARRAY_H
#define ANYDSL_ARRAY_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include <vector>

#include "anydsl/util/assert.h"

namespace anydsl {

template<class T> class Array;

//------------------------------------------------------------------------------

template<class LEFT, class RIGHT>
inline LEFT& deref_hook(RIGHT* ptr) { return *ptr; }

//------------------------------------------------------------------------------

template<class T, class U> struct dep_const { typedef U type; };
template<class T, class U> struct dep_const<const T, U> { typedef const U type; };

//------------------------------------------------------------------------------

template<class T, class Deref = T, Deref& (*Hook)(T*) = deref_hook<T, T> >
class ArrayRef {
public:

    template<class U>
    class iterator_base {
    public:

        typedef std::random_access_iterator_tag iterator_category;
        typedef typename dep_const<T, Deref>::type value_type;
        typedef ptrdiff_t difference_type;
        typedef T* pointer;
        typedef value_type& reference;

        template<class V>
        iterator_base<U>(const iterator_base<V>& i) : base_(i.base_) {}
        iterator_base<U>(T* base) : base_(base) {}

        iterator_base<U>& operator ++ () { ++base_; return *this; }
        iterator_base<U>  operator ++ (int) { iterator_base<U> i(*this); ++(*this); return i; }

        iterator_base<U>& operator -- () { --base_; return *this; }
        iterator_base<U>  operator -- (int) { iterator_base<U> i(*this); --(*this); return i; }

        difference_type operator + (iterator_base<U> i) { return difference_type(base_ + i.base()); }
        difference_type operator - (iterator_base<U> i) { return difference_type(base_ - i.base()); }

        iterator_base<U> operator + (difference_type d) { return iterator_base<U>(base_ + d); }
        iterator_base<U> operator - (difference_type d) { return iterator_base<U>(base_ - d); }

        bool operator <  (const iterator_base<U>& i) { return base_ <  i.base_; }
        bool operator <= (const iterator_base<U>& i) { return base_ <= i.base_; }
        bool operator >  (const iterator_base<U>& i) { return base_ >  i.base_; }
        bool operator >= (const iterator_base<U>& i) { return base_ >= i.base_; }
        bool operator == (const iterator_base<U>& i) { return base_ == i.base_; }
        bool operator != (const iterator_base<U>& i) { return base_ != i.base_; }

        reference operator *  () { return Hook(base_); }
        reference operator -> () { return Hook(base_); }

        T* base() const { return base_; }

    private:

        T* base_;
    };

    typedef iterator_base<T> iterator;
    typedef iterator_base<const T> const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
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

    iterator begin() { return iterator(ptr_); }
    iterator end() { return iterator(ptr_ + size_); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }

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

//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
