#ifndef ANYDSL_OPS_H
#define ANYDSL_OPS_H

namespace anydsl {

// add
#define ANYDSL_U_TYPE(T) inline T add(T a, T b) { return a + b; }
#define ANYDSL_F_TYPE(T) inline T add(T a, T b) { return a + b; }
#include "anydsl/tables/primtypetable.h"

// sub
#define ANYDSL_U_TYPE(T) inline T sub(T a, T b) { return a - b; }
#define ANYDSL_F_TYPE(T) inline T sub(T a, T b) { return a - b; }
#include "anydsl/tables/primtypetable.h"

// mul
#define ANYDSL_U_TYPE(T) inline T mul(T a, T b) { return a * b; }
#define ANYDSL_F_TYPE(T) inline T mul(T a, T b) { return a * b; }
#include "anydsl/tables/primtypetable.h"

// div
#define ANYDSL_U_TYPE(T) inline T udiv(T a, T b) { return a / b; }
#define ANYDSL_F_TYPE(T) inline T fdiv(T a, T b) { return a / b; }
#include "anydsl/tables/primtypetable.h"
#define ANYDSL_U_TYPE(T) \
    inline T sdiv(T a, T b) { \
        typedef make_signed< T >::type S; \
        return bcast< T , S>(bcast<S, T >(a) / bcast<S, T >(b)); \
    }
#include "anydsl/tables/primtypetable.h"

// rem
#define ANYDSL_U_TYPE(T) inline T urem(T a, T b) { return a / b; }
#include "anydsl/tables/primtypetable.h"
#define ANYDSL_F_TYPE(T) inline T frem(T a, T b) { return fmod(a, b); }
#include "anydsl/tables/primtypetable.h"

// and
#define ANYDSL_U_TYPE(T) inline T bit_and(T a, T b) { return a & b; }
#include "anydsl/tables/primtypetable.h"

// or
#define ANYDSL_U_TYPE(T) inline T bit_or(T a, T b) { return a | b; }
#include "anydsl/tables/primtypetable.h"

// xor
#define ANYDSL_U_TYPE(T) inline T bit_xor(T a, T b) { return a ^ b; }
#include "anydsl/tables/primtypetable.h"

// shl
#define ANYDSL_U_TYPE(T) \
    inline T shl(T a, T b) { \
        anydsl_assert(b < sizeof(T) * 8, "shift ammount out of bounds"); \
        return a << b; \
    }
#include "anydsl/tables/primtypetable.h"

// lshr
#define ANYDSL_U_TYPE(T) \
    inline T lshr(T a, T b) { \
        anydsl_assert(b < sizeof(T) * 8, "shift ammount out of bounds"); \
        return a >> b; \
    }
#include "anydsl/tables/primtypetable.h"

// ashr
#define ANYDSL_U_TYPE(T) \
    inline T ashr(T a, T b) { \
        anydsl_assert(b < sizeof(T) * 8, "shift ammount out of bounds"); \
        typedef make_signed< T >::type S; \
        return bcast< T , S>(bcast<S, T >(a) >> bcast<S, T >(b)); \
    }
#include "anydsl/tables/primtypetable.h"

// cmp_eq
#define ANYDSL_U_TYPE(T) inline u1 cmp_eq(T a, T b) { return u1(a == b); }
#include "anydsl/tables/primtypetable.h"

// cmp_ne
#define ANYDSL_U_TYPE(T) inline u1 cmp_ne(T a, T b) { return u1(a != b); }
#include "anydsl/tables/primtypetable.h"

// cmp_ugt
#define ANYDSL_U_TYPE(T) inline u1 cmp_ugt(T a, T b) { return u1(a > b); }
#include "anydsl/tables/primtypetable.h"

// cmp_uge
#define ANYDSL_U_TYPE(T) inline u1 cmp_uge(T a, T b) { return u1(a >= b); }
#include "anydsl/tables/primtypetable.h"

// cmp_ult
#define ANYDSL_U_TYPE(T) inline u1 cmp_ult(T a, T b) { return u1(a < b); }
#include "anydsl/tables/primtypetable.h"

// cmp_ule
#define ANYDSL_U_TYPE(T) inline u1 cmp_ule(T a, T b) { return u1(a <= b); }
#include "anydsl/tables/primtypetable.h"

// cmp_sgt
#define ANYDSL_U_TYPE(T) \
    inline u1 cmp_sgt(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) > bcast<S, T>(b)); \
    }
#include "anydsl/tables/primtypetable.h"

// cmp_sge
#define ANYDSL_U_TYPE(T) \
    inline u1 cmp_sge(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) >= bcast<S, T>(b)); \
    }
#include "anydsl/tables/primtypetable.h"

// cmp_slt
#define ANYDSL_U_TYPE(T) \
    inline u1 cmp_slt(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) < bcast<S, T>(b)); \
    }
#include "anydsl/tables/primtypetable.h"

// cmp_sle
#define ANYDSL_U_TYPE(T) \
    inline u1 cmp_sle(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) <= bcast<S, T>(b)); \
    }
#include "anydsl/tables/primtypetable.h"

// fcmp_oeq
#define ANYDSL_F_TYPE(T) inline u1 fcmp_oeq(T a, T b) { return u1(a == b); }
#include "anydsl/tables/primtypetable.h"

// fcmp_one
#define ANYDSL_F_TYPE(T) inline u1 fcmp_one(T a, T b) { return u1(a != b); }
#include "anydsl/tables/primtypetable.h"

// fcmp_ogt
#define ANYDSL_F_TYPE(T) inline u1 fcmp_ogt(T a, T b) { return u1(a > b); }
#include "anydsl/tables/primtypetable.h"

// fcmp_oge
#define ANYDSL_F_TYPE(T) inline u1 fcmp_oge(T a, T b) { return u1(a >= b); }
#include "anydsl/tables/primtypetable.h"

// fcmp_olt
#define ANYDSL_F_TYPE(T) inline u1 fcmp_olt(T a, T b) { return u1(a < b); }
#include "anydsl/tables/primtypetable.h"

// fcmp_ole
#define ANYDSL_F_TYPE(T) inline u1 fcmp_ole(T a, T b) { return u1(a <= b); }
#include "anydsl/tables/primtypetable.h"

// fcmp_ugt
#define ANYDSL_F_TYPE(T) inline u1 fcmp_ugt(T a, T b) { return u1(std::isgreater(a, b)); }
#include "anydsl/tables/primtypetable.h"

// fcmp_uge
#define ANYDSL_F_TYPE(T) inline u1 fcmp_uge(T a, T b) { return u1(std::isgreaterequal(a, b)); }
#include "anydsl/tables/primtypetable.h"

// fcmp_ult
#define ANYDSL_F_TYPE(T) inline u1 fcmp_ult(T a, T b) { return u1(std::isless(a, b)); }
#include "anydsl/tables/primtypetable.h"

// fcmp_ule
#define ANYDSL_F_TYPE(T) inline u1 fcmp_ule(T a, T b) { return u1(std::islessequal(a, b)); }
#include "anydsl/tables/primtypetable.h"

// fcmp_urd
#define ANYDSL_F_TYPE(T) inline u1 fcmp_urd(T a, T b) { return u1(std::isunordered(a, b)); }
#include "anydsl/tables/primtypetable.h"

// fcmp_ord
#define ANYDSL_F_TYPE(T) inline u1 fcmp_ord(T a, T b) { return u1(~fcmp_urd(a,b)); }
#include "anydsl/tables/primtypetable.h"

// fconv
inline f64 fconv(f32 f) { return f; }
inline f32 fconv(f64 f) { return f; }

//// ftos
//#define ANYDSL_U_TYPE(T) inline T ftos(f32 f) { return f; }
//#include "anydsl/tables/primtypetable.h"

#endif // ANYDSL_OPS_H

} // namespace anydsl
