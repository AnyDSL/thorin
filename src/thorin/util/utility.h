#ifndef THORIN_UTILITY_H
#define THORIN_UTILITY_H

#include <cassert>
#include <memory>
#include <stack>
#include <queue>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "thorin/util/types.h"

[[noreturn]] inline void _thorin_unreachable() { abort(); }
#define THORIN_UNREACHABLE do { assert(false && "unreachable"); ::_thorin_unreachable(); } while(0)

#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
#define THORIN_BREAK asm("int3");
#else
#define THORIN_BREAK { int* __p__ = nullptr; *__p__ = 42; }
#endif

#ifndef NDEBUG
#define assert_unused(x) assert(x)
#else
#define assert_unused(x) ((void) (0 && (x)))
#endif

namespace thorin {

/// Use to initialize an @c std::unique_ptr<T> in a lazy way.
template<class This, class T>
inline T& lazy_init(const This* self, std::unique_ptr<T>& ptr) {
    return *(ptr ? ptr : ptr = std::make_unique<T>(*self));
}

template<class S>
auto pop(S& s) -> decltype(s.top(), typename S::value_type()) {
    auto val = s.top();
    s.pop();
    return val;
}

template<class Q>
auto pop(Q& q) -> decltype(q.front(), typename Q::value_type()) {
    auto val = q.front();
    q.pop();
    return val;
}

template<class Set>
class unique_stack {
public:
    using T = typename std::remove_reference_t<Set>::value_type;

    bool push(T val) {
        if (done_.emplace(val).second) {
            stack_.emplace(val);
            return true;
        }
        return false;
    }

    bool empty() const { return stack_.empty(); }
    const T& top() { return stack_.top(); }
    T pop() { return thorin::pop(stack_); }
    void clear() { done_.clear(); stack_ = {}; }

private:
    Set done_;
    std::stack<T> stack_;
};

template<class Set>
class unique_queue {
public:
    using T = typename std::remove_reference_t<Set>::value_type;

    unique_queue() = default;
    unique_queue(Set set)
        : done_(set)
    {}

    bool push(T val) {
        if (done_.emplace(val).second) {
            queue_.emplace(val);
            return true;
        }
        return false;
    }

    bool empty() const { return queue_.empty(); }
    T pop() { return thorin::pop(queue_); }
    T& front() { return queue_.front(); }
    T& back() { return queue_.back(); }
    void clear() { done_.clear(); queue_ = {}; }

private:
    Set done_;
    std::queue<T> queue_;
};

template<class T>
struct Push {
    Push(T& t, T new_val)
        : old_(t)
        , ref_(t)
    {
        t = new_val;
    }
    ~Push() { ref_ = old_; }

private:
    T old_;
    T& ref_;
};

template<class T, class U>
inline Push<T> push(T& t, U new_val) { return Push<T>(t, new_val); }

#define THORIN_LNAME__(name, line) name##__##line
#define THORIN_LNAME_(name, line)  THORIN_LNAME__(name, line)
#define THORIN_LNAME(name)         THORIN_LNAME_(name, __LINE__)

#define THORIN_PUSH(what, with) auto THORIN_LNAME(thorin_push) = thorin::push((what), (with))

/**
 * A tagged pointer: first 16 bits is tag (index), remaining 48 bits is the actual pointer.
 * For non-x86_64 there is a fallback implementation.
 */
template<class T, class I = size_t>
class TaggedPtr {
public:
    TaggedPtr() {}
#if defined(__x86_64__) || (_M_X64)
    TaggedPtr(T* ptr, I index)
        : ptr_(reinterpret_cast<int64_t>(ptr))
        , index_(uint16_t(index))
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
    TaggedPtr& operator=(I index) { index_ = uint16_t(index); return *this; }
#else
    TaggedPtr& operator=(T*  ptr) {   ptr_ =   ptr; return *this; }
    TaggedPtr& operator=(I index) { index_ = index; return *this; }
#endif
    void set(T* ptr, I index) { (*this = ptr) = index; }

private:
#if defined(__x86_64__) || (_M_X64)
    int64_t ptr_    : 48; // sign extend to make pointer canonical
    uint64_t index_ : 16;
#else
    T* ptr_;
    I index_;
#endif
};

#if defined(__x86_64__) || (_M_X64)
static_assert(sizeof(TaggedPtr<void*,int>) == 8, "a tagged ptr on x86_64 is supposed to be 8 bytes big");
#endif

//@{ bit fiddling

/// Determines whether @p i is a power of two.
constexpr uint64_t is_power_of_2(uint64_t i) { return ((i != 0) && !(i & (i - 1))); }

constexpr uint64_t log2(uint64_t n, uint64_t p = 0) { return (n <= 1_u64) ? p : log2(n / 2_u64, p + 1_u64); }

inline uint64_t round_to_power_of_2(uint64_t i) {
    i--;
    i |= i >>  1_u64;
    i |= i >>  2_u64;
    i |= i >>  4_u64;
    i |= i >>  8_u64;
    i |= i >> 16_u64;
    i |= i >> 32_u64;
    i++;
    return i;
}

inline size_t bitcount(uint64_t v) {
#if defined(__GNUC__) | defined(__clang__)
    return __builtin_popcountll(v);
#elif defined(_MSC_VER)
    return __popcnt64(v);
#else
    // see https://stackoverflow.com/questions/3815165/how-to-implement-bitcount-using-only-bitwise-operators
    auto c = v - ((v >>  1_u64)      & 0x5555555555555555_u64);
    c =          ((c >>  2_u64)      & 0x3333333333333333_u64) + (c & 0x3333333333333333_u64);
    c =          ((c >>  4_u64) + c) & 0x0F0F0F0F0F0F0F0F_u64;
    c =          ((c >>  8_u64) + c) & 0x00FF00FF00FF00FF_u64;
    c =          ((c >> 16_u64) + c) & 0x0000FFFF0000FFFF_u64;
    return       ((c >> 32_u64) + c) & 0x00000000FFFFFFFF_u64;
#endif
}

//@}

}

#endif
