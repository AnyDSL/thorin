#ifndef THORIN_UTIL_ARRAY_H
#define THORIN_UTIL_ARRAY_H

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <vector>

#include "thorin/util/hash.h"

namespace thorin {

template<class T> class Array;

//------------------------------------------------------------------------------

template<class T>
class ArrayRef {
public:
    typedef const T* const_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    ArrayRef()
        : ptr_(nullptr)
        , size_(0)
    {}
    ArrayRef(const ArrayRef<T>& ref)
        : ptr_(ref.ptr_)
        , size_(ref.size_)
    {}
    ArrayRef(const std::vector<T>& vector)
       : ptr_(vector.data())
       , size_(vector.size())
    {}
    ArrayRef(const T* ptr, size_t size)
        : ptr_(ptr)
        , size_(size)
    {}
    ArrayRef(const Array<T>& array)
        : ptr_(array.begin())
        , size_(array.size())
    {}
    ArrayRef(std::initializer_list<T> list)
        : ptr_(list.begin())
        , size_(std::distance(list.begin(), list.end()))
    {}

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const T& operator [] (size_t i) const { assert(i < size() && "index out of bounds"); return *(ptr_ + i); }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T const& front() const { assert(!empty()); return ptr_[0]; }
    T const& back()  const { assert(!empty()); return ptr_[size_ - 1]; }
    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_to_end(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_num_from_end(size_t num) const { return ArrayRef<T>(ptr_, size_ - num); }
    ArrayRef<T> slice_from_begin(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }
    Array<T> cut(ArrayRef<size_t> indices, size_t reserve = 0) const;
    template<class Other>
    bool operator == (const Other& other) const { return this->size() == other.size() && std::equal(begin(), end(), other.begin()); }

private:
    const T* ptr_;
    size_t size_;
};

//------------------------------------------------------------------------------

template<class T>
class Array {
public:
    Array()
        : ptr_(nullptr)
        , size_(0)
    {}
    explicit Array(size_t size, const T& val = T())
        : ptr_(new T[size])
        , size_(size)
    {
        std::fill(begin(), end(), val);
    }
    Array(ArrayRef<T> ref)
        : ptr_(new T[ref.size()])
        , size_(ref.size())
    {
        std::copy(ref.begin(), ref.end(), begin());
    }
    Array(Array&& array)
        : ptr_(std::move(array.ptr_))
        , size_(std::move(array.size_))
    {
        array.ptr_ = nullptr;
        array.size_ = 0;
    }
    Array(const Array& array)
        : ptr_(new T[array.size()])
        , size_(array.size())
    {
        std::copy(array.begin(), array.end(), begin());
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

    T& operator [] (size_t i) { assert(i < size() && "index out of bounds"); return ptr_[i]; }
    T const& operator [] (size_t i) const { assert(i < size() && "index out of bounds"); return ptr_[i]; }
    T& front() const { assert(!empty()); return ptr_[0]; }
    T& back()  const { assert(!empty()); return ptr_[size_ - 1]; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_to_end(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_num_from_end(size_t num) const { return ArrayRef<T>(ptr_, size_ - num); }
    ArrayRef<T> slice_from_begin(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }
    Array<T> cut(ArrayRef<size_t> indices, size_t reserve = 0) const { return ArrayRef<T>(*this).cut(indices, reserve); }
    bool operator == (const Array<T>& other) const { return ArrayRef<T>(*this) == ArrayRef<T>(other); }
    void shrink(size_t newsize) { assert(newsize <= size_); size_ = newsize; }
    ArrayRef<T> ref() const { return ArrayRef<T>(ptr_, size_); }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    friend void swap(Array& a, Array& b) {
        using std::swap;
        swap(a.ptr_,  b.ptr_);
        swap(a.size_, b.size_);
    }
    Array<T>& operator= (Array<T> other) { swap(*this, other); return *this; }

private:
    T* ptr_;
    size_t size_;
};

template<class T>
Array<T> ArrayRef<T>::cut(ArrayRef<size_t> indices, size_t reserve) const {
    size_t num_old = size();
    size_t num_idx = indices.size();
    size_t num_res = num_old - num_idx;

    Array<T> result(num_res + reserve);

    for (size_t o = 0, i = 0, r = 0; o < num_old; ++o) {
        if (i < num_idx && indices[i] == o)
            ++i;
        else
            result[r++] = (*this)[o];
    }

    return result;
}

//------------------------------------------------------------------------------

template<class T>
inline size_t hash_combine(size_t seed, thorin::ArrayRef<T> aref) {
    for (size_t i = 0, e = aref.size(); i != e; ++i)
        seed = hash_combine(seed, aref[i]);
    return seed;
}

//------------------------------------------------------------------------------

template<class T>
struct Hash<thorin::ArrayRef<T>> {
    size_t operator () (thorin::ArrayRef<T> aref) const { return hash_combine(0, aref); }
};

template<class T>
struct Hash<thorin::Array<T>> {
    size_t operator () (const thorin::Array<T>& array) const { return hash_value(array.ref()); }
};

}

#endif
