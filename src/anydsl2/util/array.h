#ifndef ANYDSL2_ARRAY_H
#define ANYDSL2_ARRAY_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include <vector>

#include "anydsl2/util/assert.h"
#include "anydsl2/util/hash.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

template<class T> class Array;

//------------------------------------------------------------------------------

template<class T>
class ArrayRef {
public:

    typedef const T* const_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    ArrayRef()
        : ptr_(0)
        , size_(0)
    {}
    ArrayRef(const ArrayRef<T>& ref)
        : ptr_(ref.ptr_)
        , size_(ref.size_)
    {}
    template<size_t N>
    ArrayRef(T (&array)[N])
        : ptr_(&array[0])
        , size_(N)
    {}
    ArrayRef(const std::vector<T>& vector)
       : ptr_(vector.empty() ? 0 : &vector.front())
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

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    const T& operator [] (size_t i) const {
        assert(i < size() && "index out of bounds");
        return *(ptr_ + i);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T const& front() const { assert(!empty()); return ptr_[0]; }
    T const& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_front(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_back(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }
    Array<T> cut(ArrayRef<size_t> indices, size_t reserve = 0) const;

    bool operator == (ArrayRef<T> other) const {
        if (size() != other.size())
            return false;

        for (size_t i = 0, e = size(); i != e; ++i)
            if (!(ptr_[i] == other[i]))
                return false;

        return true;
    }

    template<class U>
    ArrayRef<U> cast() const { return ArrayRef<U>((const U*) ptr_, size_); }

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
        std::copy(ref.begin(), ref.end(), begin());
    }
    explicit Array(const Array<T>& array)
        : ptr_(new T[array.size()])
        , size_(array.size())
    {
        std::copy(array.begin(), array.end(), begin());
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
    Array<T> cut(ArrayRef<size_t> indices, size_t reserve = 0) const { return ArrayRef<T>(*this).cut(indices, reserve); }

    bool operator == (const Array<T>& other) const { return ArrayRef<T>(*this) == ArrayRef<T>(other); }

    void shrink(size_t newsize) { assert(newsize <= size_); size_ = newsize; }

    ArrayRef<T> ref() const { return ArrayRef<T>(ptr_, size_); }

private:

    Array<T>& operator = (const Array<T>& array);

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
inline size_t hash_combine(size_t seed, anydsl2::ArrayRef<T> aref) {
    for (size_t i = 0, e = aref.size(); i != e; ++i)
        seed = hash_combine(seed, aref[i]);
    return seed;
}

template<class T> inline size_t hash_value(ArrayRef<T> aref) { return hash_combine(0, aref); }
template<class T> inline size_t hash_value(const Array<T>& array) { return hash_value(array.ref()); }

//------------------------------------------------------------------------------

typedef boost::tuples::null_type bo_null_type;

template<class T> inline bool is_empty(T) { return false; }
template<class T> inline bool is_empty(ArrayRef<T> array) { return array.empty(); }

                  inline bool t_is_empty(bo_null_type) { return true; }
template<class T> inline bool t_is_empty(const T& t) { return is_empty(t.get_head()) && t_is_empty(t.get_tail()); }

template<class TH,           class UH, class UT> inline bool t_smart_eq(TH th, bo_null_type, UH uh, const UT& ut) { return th == uh && t_is_empty(ut); } 
template<class TH, class TT, class UH          > inline bool t_smart_eq(TH th, const TT& tt, UH uh, bo_null_type) { return false; }
template<class TH                              > inline bool t_smart_eq(TH th, bo_null_type, TH uh, bo_null_type) { return th == uh; } 
template<class TH, class TT, class UH, class UT> inline bool t_smart_eq(TH th, const TT& tt, UH uh, const UT& ut) { return th == uh && t_smart_eq(tt.get_head(), tt.get_tail(), ut.get_head(), ut.get_tail()); }

template<class TH, class TT, class UT> inline bool t_smart_eq(TH th, const TT& tt, ArrayRef<TH> array, const UT& ut) { return a_smart_eq(th, tt, array, 0, ut); }
template<class TH, class TT>           inline bool t_smart_eq(TH th, const TT& tt, ArrayRef<TH> array, bo_null_type) { return a_smart_eq(th, tt, array, 0, bo_null_type()); }
template<class TH>                     inline bool t_smart_eq(TH th, bo_null_type, ArrayRef<TH> array, bo_null_type) { return a_smart_eq(th, bo_null_type(), array, 0, bo_null_type()); }

template<class TH> inline bool
a_smart_eq(TH th, bo_null_type, ArrayRef<TH> array, size_t i, bo_null_type) {
    return i+1 == array.size() ? th == array[i] : false;
}
template<class TH, class TT> inline bool
a_smart_eq(TH th, const TT& tt, ArrayRef<TH> array, size_t i, bo_null_type) {
    if (i < array.size())
        return th == array[i] && a_smart_eq(tt.get_head(), tt.get_tail(), array, i+1, bo_null_type());
    else
        return false;
}
template<class TH, class TT, class UH, class UT> inline bool
a_smart_eq(TH th, const TT& tt, ArrayRef<TH> array, size_t i, const UT& ut) {
    if (i < array.size())
        return th == array[i] && a_smart_eq(tt.get_head(), tt.get_tail(), array, i+1, ut);
    else
        return t_smart_eq(th, tt, ut.get_head(), ut.get_tail());
}

template<class T, class U> inline bool
smart_eq(const T& t, const U& u) { return t_smart_eq(t.get_head(), t.get_tail(), u.get_head(), u.get_tail()); }

} // namespace anydsl2

#endif
