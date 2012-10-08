#ifndef ANYDSL2_UTIL_TYPES_H
#define ANYDSL2_UTIL_TYPES_H

#include <cmath>
#include <ostream>

#include <boost/cstdint.hpp>

#include "anydsl2/util/cast.h"

namespace anydsl2 {

typedef  int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef  uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float    f32;
typedef double   f64;

class u1 {
public:

    u1() {}
    u1(bool b) : b_(b) {}
    u1( i8 i) : b_(i & 1) {}
    u1(i16 i) : b_(i & 1) {}
    u1(i32 i) : b_(i & 1) {}
    u1(i64 i) : b_(i & 1ll) {}
    u1( u8 u) : b_(u & 1u) {}
    u1(u16 u) : b_(u & 1u) {}
    u1(u32 u) : b_(u & 1u) {}
    u1(u64 u) : b_(u & 1ull) {}

    u1 operator + (u1 u) { return u1(get() + u.get()); }
    u1 operator - (u1 u) { return u1(get() - u.get()); }
    u1 operator * (u1 u) { return u1(get() * u.get()); }
    u1 operator / (u1 u) { return u1(get() / u.get()); }
    u1 operator % (u1 u) { return u1(get() % u.get()); }
    u1 operator & (u1 u) { return u1(get() & u.get()); }
    u1 operator | (u1 u) { return u1(get() | u.get()); }
    u1 operator ^ (u1 u) { return u1(get() ^ u.get()); }

    u1 operator >>(u1 u) { return u1(get() >> u.get()); } // TODO correct?
    u1 operator <<(u1 u) { return u1(get() << u.get()); } // TODO correct?

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

private:

    bool b_;
};

// TODO
typedef u1 i1;

template<class T> struct make_signed {};
template<> struct make_signed< u1> { typedef  i1 type; };
template<> struct make_signed< u8> { typedef  i8 type; };
template<> struct make_signed<u16> { typedef i16 type; };
template<> struct make_signed<u32> { typedef i32 type; };
template<> struct make_signed<u64> { typedef i64 type; };

template<class T> struct is_u_type { static const bool value = false; };
#define ANYDSL2_JUST_U_TYPE(T) template<> struct is_u_type<T> { static const bool value = true; };
#include "anydsl2/tables/primtypetable.h"

template<class T> struct is_f_type { static const bool value = false; };
#define ANYDSL2_JUST_F_TYPE(T) template<> struct is_u_type<T> { static const bool value = true; };
#include "anydsl2/tables/primtypetable.h"


inline std::ostream& operator << (std::ostream& o, u1 u) { return o << ((unsigned) u.get()); }

} // namespace anydsl2

#endif // ANYDSL2_UTIL_TYPES_H
