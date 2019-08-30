#ifndef THORIN_UTIL_TYPES_H
#define THORIN_UTIL_TYPES_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif
#define HALF_ROUND_STYLE 1
#define HALF_ROUND_TIES_TO_EVEN 1
#include <half.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace thorin {

using half_float::half;

#define THORIN_8_16_32_64(m) m(8) m(16) m(32) m(64)
#define THORIN_16_32_64(m)        m(16) m(32) m(64)

template<class T>
bool get_sign(T val) {
    static_assert(std::is_integral<T>(), "get_sign only supported for signed and unsigned integer types");

    if constexpr(std::is_integral<T>()) {
        if constexpr(std::is_signed<T>())
            return val < 0;
        else
            return val >> (T(sizeof(val)) * T(8) - T(1));
    }
}

template<int> struct w2u_ {};
template<int> struct w2s_ {};
template<int> struct w2r_ {};

#define CODE(i) \
    typedef  int ## i ##_t s ## i; \
    typedef uint ## i ##_t u ## i; \
    template<> struct w2u_<i> { typedef u ## i type; }; \
    template<> struct w2s_<i> { typedef s ## i type; }; \
    constexpr s ## i operator"" _s ## i(unsigned long long int s) { return s ## i(s); } \
    constexpr u ## i operator"" _u ## i(unsigned long long int u) { return u ## i(u); }
THORIN_8_16_32_64(CODE)
#undef CODE

// Map both signed 1 and unsigned 1 to bool
template<> struct w2u_<1> { typedef bool type; };
template<> struct w2s_<1> { typedef bool type; };

typedef half   r16;
typedef float  r32;
typedef double r64;

#define CODE(i) \
    template<> struct w2r_<i> { typedef r ## i type; };
THORIN_16_32_64(CODE)
#undef CODE

template<int w> using w2u = typename w2u_<w>::type;
template<int w> using w2s = typename w2s_<w>::type;
template<int w> using w2r = typename w2r_<w>::type;

/// A @c size_t literal. Use @c 0_s to disambiguate @c 0 from @c nullptr.
constexpr size_t operator""_s(unsigned long long int i) { return size_t(i); }
inline /*constexpr*/ r16 operator""_f16(long double d) { return r16(d); } // wait till fixed upstream
constexpr r32 operator""_r32(long double d) { return r32(d); }
constexpr r64 operator""_r64(long double d) { return r64(d); }

}

#endif
