#ifndef THORIN_UTIL_PTR_H
#define THORIN_UTIL_PTR_H

#include <cstdint>
#include <memory>

namespace thorin {

/// Use to initialize an @c std::unique_ptr<T> in a lazy way.
template<class This, class T>
inline T& lazy_init(const This* self, std::unique_ptr<T>& ptr) {
    return *(ptr ? ptr : ptr = std::make_unique<T>(*self));
}

/// A tagged pointer: first 16 bits is tag (index), remaining 48 bits is the actual pointer.
/// For non-x86_64 there is a fallback implementation.
template<class T, class I = size_t>
class TaggedPtr {
public:
    TaggedPtr() {}
#if defined(__x86_64__) || (_M_X64)
    TaggedPtr(T* ptr, I index)
        : ptr_(reinterpret_cast<int64_t>(ptr))
        , index_(int16_t(index))
    {}
#else
    TaggedPtr(T* ptr, I index)
        : ptr_(ptr)
        , index_(I(index))
    {}
#endif

    T* ptr() const { return reinterpret_cast<T*>(ptr_); }
    T* operator->() const { return ptr(); }
    operator T*() const { return ptr(); }
    void index(I index) { index_ = index; }
    I index() const { return I(index_); }
    bool operator==(TaggedPtr other) const { return this->ptr() == other.ptr() && this->index() == other.index(); }

#if defined(__x86_64__) || (_M_X64)
    TaggedPtr& operator=(T*  ptr) { ptr_ = reinterpret_cast<int64_t>(ptr); return *this; }
    TaggedPtr& operator=(I index) { index_ = int16_t(index); return *this; }
#else
    TaggedPtr& operator=(T*  ptr) {   ptr_ =   ptr; return *this; }
    TaggedPtr& operator=(I index) { index_ = index; return *this; }
#endif
    void set(T* ptr, I index) { (*this = ptr) = index; }

private:
#if defined(__x86_64__) || (_M_X64)
    int64_t ptr_    : 48; // sign extend to make pointer canonical
    int64_t index_  : 16;
#else
    T* ptr_;
    I index_;
#endif
};

#if defined(__x86_64__) || (_M_X64)
static_assert(sizeof(TaggedPtr<void*,int>) == 8, "a tagged ptr on x86_64 is supposed to be 8 bytes big");
#endif

}

#endif
