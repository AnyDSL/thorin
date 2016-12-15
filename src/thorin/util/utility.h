#ifndef THORIN_UTILITY_H
#define THORIN_UTILITY_H

#include <cassert>
#include <memory>
#include <queue>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifndef _MSC_VER
#define THORIN_UNREACHABLE do { assert(true && "unreachable"); abort(); } while(0)
#else // _MSC_VER
inline __declspec(noreturn) void thorin_dummy_function() { abort(); }
#define THORIN_UNREACHABLE do { assert(true && "unreachable"); thorin_dummy_function(); } while(0)
#endif

#ifndef NDEBUG
#define THORIN_CALL_ONCE do { static bool once = true; assert(once); once=false; } while(0)
#define assert_unused(x) assert(x)
#else
#define THORIN_CALL_ONCE
#define assert_unused(x) ((void) (0 && (x)))
#endif

#define THORIN_IMPLIES(a, b) (!(a) || ((a) && (b)))

// http://stackoverflow.com/questions/1489932/how-to-concatenate-twice-with-the-c-preprocessor-and-expand-a-macro-as-in-arg
#define THORIN_PASTER(x,y) x ## y

namespace thorin {

/// Use to initialize an \p AutoPtr in a lazy way.
template<class This, class T>
inline T& lazy_init(const This* self, std::unique_ptr<T>& ptr) {
    return *(ptr ? ptr : (ptr.reset(new T(*self)), ptr));
}

template<class T>
T pop(std::queue<T>& queue) {
    auto val = queue.front();
    queue.pop();
    return val;
}

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
 * A tagged pointer: first 16 bits is tag, remaining 48 bits is the actual pointer.
 * For non-x86_64 there is a fallback implementation.
 */
template<class T>
class TaggedPtr {
public:
    TaggedPtr() {}
#if defined(__x86_64__) || (_M_X64)
    TaggedPtr(uint16_t tag, T* ptr)
        : tag_(tag)
        , ptr_(reinterpret_cast<int64_t>(ptr))
    {}
#else
    TaggedPtr(uint16_t tag, T* ptr)
        : tag_(tag)
        , ptr_(ptr)
    {}
#endif

    T* ptr() const { return reinterpret_cast<T*>(ptr_); }
    T* operator->() const { return ptr(); }
    operator T*() const { return ptr(); }
    void tag(uint16_t tag) { tag_ = tag; }
    uint16_t tag() const { return tag_; }
    bool operator==(TaggedPtr other) const { return this->ptr() == other.ptr() && this->tag() == other.tag(); }

private:
    uint16_t tag_;
#if defined(__x86_64__) || (_M_X64)
    int64_t ptr_ : 48; // sign extend to make pointer canonical
#else
    T* ptr_;
#endif
};

//@{ bit fiddling

/// Determines whether @p i is a power of two.
constexpr size_t is_power_of_2(size_t i) { return ((i != 0) && !(i & (i - 1))); }

constexpr unsigned log2(unsigned n, unsigned p = 0) { return (n <= 1) ? p : log2(n / 2, p + 1); }

constexpr uint32_t round_to_power_of_2(uint32_t i) {
    i--;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
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
    auto c = v - ((v >>  1ull)      & 0x5555555555555555ull);
    c =          ((c >>  2ull)      & 0x3333333333333333ull) + (c & 0x3333333333333333ull);
    c =          ((c >>  4ull) + c) & 0x0F0F0F0F0F0F0F0Full;
    c =          ((c >>  8ull) + c) & 0x00FF00FF00FF00FFull;
    c =          ((c >> 16ull) + c) & 0x0000FFFF0000FFFFull;
    return       ((c >> 32ull) + c) & 0x00000000FFFFFFFFull;
#endif
}

//@}

}

#endif
