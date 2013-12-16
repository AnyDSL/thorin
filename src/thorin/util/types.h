#ifndef THORIN_UTIL_TYPES_H
#define THORIN_UTIL_TYPES_H

#include <cmath>
#include <cstdint>
#include <ostream>

#include "thorin/util/cast.h"

namespace thorin {

class u1 {
public:
    u1() {}
    u1(    bool b) : b_(b) {}
    u1(  int8_t i) : b_(i & 1) {}
    u1( int16_t i) : b_(i & 1) {}
    u1( int32_t i) : b_(i & 1) {}
    u1( int64_t i) : b_(i & 1ll) {}
    u1( uint8_t i) : b_(i & 1u) {}
    u1(uint16_t i) : b_(i & 1u) {}
    u1(uint32_t i) : b_(i & 1u) {}
    u1(uint64_t i) : b_(i & 1ull) {}

    u1 operator + (u1 u) { return u1(get() + u.get()); }
    u1 operator - (u1 u) { return u1(get() - u.get()); }
    u1 operator * (u1 u) { return u1(get() * u.get()); }
    u1 operator & (u1 u) { return u1(get() & u.get()); }
    u1 operator | (u1 u) { return u1(get() | u.get()); }
    u1 operator ^ (u1 u) { return u1(get() ^ u.get()); }

    /**
     * \verbatim
        this | u || result
        ------------------
           0 | 0 || assert
           0 | 1 || 0
           1 | 0 || assert
           1 | 1 || 1
    \endverbatim
    */
    u1 operator / (u1 u) { assert(u.get()); return u1(this->get()); }

    /**
     * \verbatim
        this | u || result
        ------------------
           0 | 0 || assert
           0 | 1 || 0
           1 | 0 || assert
           1 | 1 || 0
    \endverbatim
    */
    u1 operator % (u1 u) { assert(u.get()); return u1(0); }

    /**
     * \verbatim
        this | u || result
        ------------------
           0 | 0 || 0
           0 | 1 || 0
           1 | 0 || 1
           1 | 1 || 0
    \endverbatim
    */
    u1 operator >> (u1 u) { return u1(this->get() && !u.get()); }
    
    /**
     * \verbatim
        this | u || result
        ------------------
           0 | 0 || 0
           0 | 1 || 0
           1 | 0 || 1
           1 | 1 || 0
    \endverbatim
    */
    u1 operator << (u1 u) { return u1(this->get() && !u.get()); }

    bool operator == (u1 u) { return get() == u.get(); }
    bool operator != (u1 u) { return get() != u.get(); }
    bool operator  < (u1 u) { return get()  < u.get(); }
    bool operator <= (u1 u) { return get() <= u.get(); }
    bool operator  > (u1 u) { return get()  > u.get(); }
    bool operator >= (u1 u) { return get() >= u.get(); }

    u1& operator += (u1 u) { b_ = (*this + u1(u)).get(); return *this; }
    u1& operator -= (u1 u) { b_ = (*this - u1(u)).get(); return *this; }
    u1& operator *= (u1 u) { b_ = (*this * u1(u)).get(); return *this; }
    u1& operator /= (u1 u) { b_ = (*this / u1(u)).get(); return *this; }
    u1& operator %= (u1 u) { b_ = (*this % u1(u)).get(); return *this; }
    u1& operator &= (u1 u) { b_ = (*this & u1(u)).get(); return *this; }
    u1& operator |= (u1 u) { b_ = (*this | u1(u)).get(); return *this; }
    u1& operator ^= (u1 u) { b_ = (*this ^ u1(u)).get(); return *this; }

    u1 operator ! () { return u1(!b_); }
    u1 operator ~ () { return u1(!b_); }

    // pre
    u1& operator ++ () { b_ = (*this + u1(true)).get(); return *this; }
    u1& operator -- () { b_ = (*this - u1(true)).get(); return *this; }

    // post
    u1 operator ++ (int) { u1 tmp = *this; b_ = (*this + u1(true)).get(); return tmp; }
    u1 operator -- (int) { u1 tmp = *this; b_ = (*this - u1(true)).get(); return tmp; }

    bool get() const { return b_; }

    operator uint32_t() { return b_; }

private:
    bool b_;
};

typedef      u1  s1;                       typedef      u1  ps1; typedef       u1  pu1; typedef      u1  qs1; typedef       u1  qu1;
typedef  int8_t  s8; typedef  uint8_t  u8; typedef  int8_t  ps8; typedef  uint8_t  pu8; typedef  int8_t  qs8; typedef  uint8_t  qu8;
typedef int16_t s16; typedef uint16_t u16; typedef int16_t ps16; typedef uint16_t pu16; typedef int16_t qs16; typedef uint16_t qu16;
typedef int32_t s32; typedef uint32_t u32; typedef int32_t ps32; typedef uint32_t pu32; typedef int32_t qs32; typedef uint32_t qu32;
typedef int64_t s64; typedef uint64_t u64; typedef int64_t ps64; typedef uint64_t pu64; typedef int64_t qs64; typedef uint64_t qu64;

typedef float  f32; typedef float  pf32;typedef float  qf32;
typedef double f64; typedef double pf64;typedef double qf64;

inline std::ostream& operator << (std::ostream& o, u1 u) { return o << ((unsigned) u.get()); }

}

#endif
