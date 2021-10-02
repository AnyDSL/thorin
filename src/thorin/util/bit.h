#ifndef THORIN_UTIL_BIT_H
#define THORIN_UTIL_BIT_H

#include <cstdint>

#include "thorin/util/types.h"

namespace thorin {

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

inline u64 pad(u64 offset, u64 align) {
    auto mod = offset % align;
    if (mod != 0) offset += align - mod;
    return offset;
}

}

#endif
