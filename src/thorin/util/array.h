#ifndef THORIN_UTIL_ARRAY_H
#define THORIN_UTIL_ARRAY_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <functional>
#include <vector>

#include "thorin/util/stream.h"

namespace thorin {

template<class T>
class Array;

//------------------------------------------------------------------------------

/**
 * A container-like wrapper for an array.
 * The array may either stem from a C array, a <tt>std::vector</tt>, a <tt>std::initializer_list</tt>, an @p Array or another @p ArrayRef.
 * @p ArrayRef does <em>not</em> own the data and, thus, does not destroy any data.
 * Likewise, you must be carefull to not destroy data an @p ArrayRef is pointing to.
 * Thorin makes use of @p ArrayRef%s in many places.
 * Note that you can often construct an @p ArrayRef inline with an initializer_list: <code>foo(arg1, {elem1, elem2, elem3}, arg3)</code>.
 * Useful operations are @p skip_front and @p skip_back to create other @p ArrayRef%s.
 */
template<class T>
class ArrayRef {
public:
    typedef T value_type;
    typedef const T* const_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    ArrayRef()
        : size_(0)
        , ptr_(nullptr)
    {}
    ArrayRef(const T* ptr, size_t size)
        : size_(size)
        , ptr_(ptr)
    {}
    template<size_t N>
    ArrayRef(const T (&ptr)[N])
        : size_(N)
        , ptr_(ptr)
    {}
    ArrayRef(const ArrayRef<T>& ref)
        : size_(ref.size_)
        , ptr_(ref.ptr_)
    {}
    ArrayRef(const Array<T>& array)
        : size_(array.size())
        , ptr_(array.begin())
    {}
    template<size_t N>
    ArrayRef(const std::array<T, N>& array)
        : size_(N)
        , ptr_(array.data())
    {}
    ArrayRef(std::initializer_list<T> list)
        : size_(std::distance(list.begin(), list.end()))
        , ptr_(list.begin())
    {}
    ArrayRef(const std::vector<T>& vector)
       : size_(vector.size())
       , ptr_(vector.data())
    {}

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const T& operator[](size_t i) const { assert(i < size() && "index out of bounds"); return *(ptr_ + i); }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T const& front() const { assert(!empty()); return ptr_[0]; }
    T const& back()  const { assert(!empty()); return ptr_[size_ - 1]; }
    ArrayRef<T> skip_front(size_t num = 1) const { return ArrayRef<T>(ptr_ + num, size() - num); }
    ArrayRef<T> skip_back (size_t num = 1) const { return ArrayRef<T>(ptr_, size() - num); }
    ArrayRef<T> get_front (size_t num = 1) const { assert(num <= size()); return ArrayRef<T>(ptr_, num); }
    ArrayRef<T> get_back  (size_t num = 1) const { assert(num <= size()); return ArrayRef<T>(ptr_ + size() - num, num); }
    Array<T> cut(ArrayRef<size_t> indices, size_t reserve = 0) const;
    template<class Other>
    bool operator==(const Other& other) const { return this->size() == other.size() && std::equal(begin(), end(), other.begin()); }
    void dump() const { stream(std::cout) << "\n"; }
    std::ostream& stream(std::ostream& os) const {
        return stream_list(os, *this, [&] (const auto& elem) { os << elem; }, "{", "}");
    }

private:
    size_t size_;
    const T* ptr_;
};

template<class T>
std::ostream& operator<<(std::ostream& os, const ArrayRef<T> a) {
    return a.stream(os);
}

//------------------------------------------------------------------------------


/**
 * A container for a heap-allocated array.
 * This class is similar to <tt>std::vector</tt> with the following differences:
 *  - In contrast to std::vector, Array cannot grow dynamically.
 *    An @p Array may @p shrink, however.
 *    But once shrunk, there is no way back.
 *  - Because of this @p Array is slightly more lightweight and usually consumes slightly less memory than <tt>std::vector</tt>.
 *  - @p Array integrates nicely with the usefull @p ArrayRef container.
 */
template<class T>
class Array {
public:
    typedef T value_type;
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    Array()
        : size_(0)
        , ptr_(nullptr)
    {}
    explicit Array(size_t size)
        : size_(size)
        , ptr_(new T[size]())
    {}
    Array(size_t size, const T& val)
        : size_(size)
        , ptr_(new T[size])
    {
        std::fill(begin(), end(), val);
    }
    Array(ArrayRef<T> ref)
        : size_(ref.size())
        , ptr_(new T[ref.size()])
    {
        std::copy(ref.begin(), ref.end(), this->begin());
    }
    Array(Array&& other)
        : size_(std::move(other.size_))
        , ptr_(std::move(other.ptr_))
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    Array(const Array& other)
        : size_(other.size())
        , ptr_(new T[other.size()])
    {
        std::copy(other.begin(), other.end(), this->begin());
    }
    Array(const std::vector<T>& other)
        : size_(other.size())
        , ptr_(new T[other.size()])
    {
        std::copy(other.begin(), other.end(), this->begin());
    }
    template<class I>
    Array(const I begin, const I end)
        : size_(std::distance(begin, end))
        , ptr_(new T[size_])
    {
        std::copy(begin, end, ptr_);
    }
    Array(std::initializer_list<T> list)
        : size_(std::distance(list.begin(), list.end()))
        , ptr_(new T[size_])
    {
        std::copy(list.begin(), list.end(), ptr_);
    }
    Array(size_t size, std::function<T(size_t)> f)
        : Array(size)
    {
        for (size_t i = 0; i != size; ++i)
            (*this)[i] = f(i);
    }

    ~Array() { delete[] ptr_; }

    iterator begin() { return ptr_; }
    iterator end() { return ptr_ + size_; }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    T& front() const { assert(!empty()); return ptr_[0]; }
    T& back()  const { assert(!empty()); return ptr_[size_ - 1]; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    ArrayRef<T> skip_front(size_t num = 1) const { return ArrayRef<T>(ptr_ + num, size() - num); }
    ArrayRef<T> skip_back (size_t num = 1) const { return ArrayRef<T>(ptr_, size() - num); }
    ArrayRef<T> get_front (size_t num = 1) const { assert(num <= size()); return ArrayRef<T>(ptr_, num); }
    ArrayRef<T> get_back  (size_t num = 1) const { assert(num <= size()); return ArrayRef<T>(ptr_ + size() - num, num); }
    Array<T> cut(ArrayRef<size_t> indices, size_t reserve = 0) const { return ArrayRef<T>(*this).cut(indices, reserve); }
    void shrink(size_t newsize) { assert(newsize <= size_); size_ = newsize; }
    ArrayRef<T> ref() const { return ArrayRef<T>(ptr_, size_); }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    T& operator[](size_t i) { assert(i < size() && "index out of bounds"); return ptr_[i]; }
    T const& operator[](size_t i) const { assert(i < size() && "index out of bounds"); return ptr_[i]; }
    bool operator==(const Array<T>& other) const { return ArrayRef<T>(*this) == ArrayRef<T>(other); }
    Array<T>& operator=(Array<T> other) { swap(*this, other); return *this; }
    void dump() const { ref().dump(); }

    friend void swap(Array& a, Array& b) {
        using std::swap;
        swap(a.size_, b.size_);
        swap(a.ptr_,  b.ptr_);
    }

private:
    size_t size_;
    T* ptr_;
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

template<class T, class U>
auto concat(const T& a, const U& b) -> Array<typename T::value_type> {
    Array<typename T::value_type> result(a.size() + b.size());
    std::copy(b.begin(), b.end(), std::copy(a.begin(), a.end(), result.begin()));
    return result;
}

template<class T>
auto concat(const T& val, ArrayRef<T> a) -> Array<T> {
    Array<T> result(a.size() + 1);
    std::copy(a.begin(), a.end(), result.begin()+1);
    result.front() = val;
    return result;
}

template<class T>
auto concat(ArrayRef<T> a, const T& val) -> Array<T> {
    Array<T> result(a.size() + 1);
    std::copy(a.begin(), a.end(), result.begin());
    result.back() = val;
    return result;
}

template<class T>
Array<typename T::value_type> make_array(const T& container) {
    return Array<typename T::value_type>(container.begin(), container.end());
}

}

#endif
