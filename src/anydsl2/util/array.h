#ifndef ANYDSL_ARRAY_H
#define ANYDSL_ARRAY_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include <vector>

#include <boost/functional/hash.hpp>

#include "anydsl2/util/assert.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

template<class T> class Array;

//------------------------------------------------------------------------------

template<class LEFT, class RIGHT>
inline LEFT const& deref_hook(const RIGHT* ptr) { return *ptr; }

//------------------------------------------------------------------------------

template<class T, class U> struct dep_const { typedef U type; };
template<class T, class U> struct dep_const<const T, U> { typedef const U type; };

//------------------------------------------------------------------------------

template<class T, class Deref = T, Deref const& (*Hook)(const T*) = deref_hook<T, T> >
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
        const_iterator(pointer base) : base_(base) {}

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

        reference operator *  () { return Hook(base_); }
        reference operator -> () { return Hook(base_); }

        pointer base() const { return base_; }

    private:

        pointer base_;
    };

    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    template<size_t N>
    ArrayRef(T (&array)[N])
        : ptr_(&array[0])
        , size_(N)
    {}
    ArrayRef(const std::vector<T>& vector)
        : ptr_(&*vector.begin())
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
    template<class Deref_, Deref const& (*Hook_)(const T*)>
    ArrayRef(const ArrayRef<T, Deref_, Hook_>& array)
        : ptr_(array.begin().base())
        , size_(array.size())
    {}

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    Deref const& operator [] (size_t i) const {
        assert(i < size() && "index out of bounds");
        return Hook(ptr_ + i);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T const& front() const { assert(!empty()); return ptr_[0]; }
    T const& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

    ArrayRef<T, Deref, Hook> slice(size_t begin, size_t end) const { return ArrayRef<T, Deref, Hook>(ptr_ + begin, end - begin); }
    ArrayRef<T, Deref, Hook> slice_front(size_t end) const { return ArrayRef<T, Deref, Hook>(ptr_, end); }
    ArrayRef<T, Deref, Hook> slice_back(size_t begin) const { return ArrayRef<T, Deref, Hook>(ptr_ + begin, size_ - begin); }

    //operator ArrayRef<T, Deref, Hook> () { return ArrayRef<T, Deref, Hook>(ptr_, size_); }
    bool operator == (ArrayRef<T, Deref, Hook> other) const {
        if (size() != other.size())
            return false;

        bool result = true;
        for (size_t i = 0, e = size(); i != e && result; ++i)
            result &= ptr_[i] == other[i];

        return result;
    }

private:

    const T* ptr_;
    size_t size_;
};

//------------------------------------------------------------------------------

template<class T>
class Array {
public:

    Array()
        : ptr_(0)
        , size_(0)
    {}
    explicit Array(size_t size)
        : ptr_(new T[size]())
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

    void alloc(size_t size) {
        assert(ptr_ == 0 && size_ == 0);
        ptr_ = new T[size]();
        size_ = size;
    };

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

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    bool valid() const { return ptr_; }

    T& front() const { assert(!empty()); return ptr_[0]; }
    T& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_front(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_back(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }

    bool operator == (const Array<T>& other) const { return ArrayRef<T>(*this) == ArrayRef<T>(other); }

    void shrink(size_t newsize) { assert(newsize <= size_); size_ = newsize; }

private:

    Array<T>& operator = (const Array<T>& array);

    T* ptr_;
    size_t size_;
};

template<class T>
inline size_t hash_value(ArrayRef<T> aref) {
    size_t seed = 0;
    boost::hash_combine(seed, aref.size());

    for (size_t i = 0, e = aref.size(); i != e; ++i)
        boost::hash_combine(seed, aref[i]);

    return seed;
}

template<class T>
inline size_t hash_value(const Array<T>& array) { return hash_value(ArrayRef<T>(array)); }

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
