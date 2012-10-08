#ifndef ANYDSL2_OPS_H
#define ANYDSL2_OPS_H

#include <boost/static_assert.hpp>

#include "anydsl2/util/types.h"

namespace anydsl2 {

/*
 * arithops -- integer operations
 */

template<class T>
inline T op_add(bool& bot, T a, T b) { BOOST_STATIC_ASSERT(is_u_type<T>::value); bot = false; return a + b; }

template<class T>
inline T op_sub(bool& bot, T a, T b) { bot = false; return a - b; }

template<class T>
inline T op_mul(bool& bot, T a, T b) { bot = false; return a * b; }

template<class T>
inline T op_udiv(bool& bot, T a, T b) {
    if (b) {
        bot = false;
        return a / b;
    } else {
        bot = true;
        return T(0);
    }
}

#if 0

template<class T>
inline T arithop_urem(bool bot&, T a, T b) { \
    if (b) { \
        bot = false; \
        return a % b; \
    } else { \
        bot = true;
        return T(0);
    } \
}

#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_U_TYPE(T) inline T op_and(bool bot&, T a, T b) { bot = false; return a & b; }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_U_TYPE(T) inline T op_or(bool bot&, T a, T b) { bot = false; return a | b; }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_U_TYPE(T) inline T op_xor(bool bot&, T a, T b) { bot = false; return a ^ b; }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_U_TYPE(T) \
    inline T op_shl(T a, T b) { \
        anydsl_assert(b < sizeof(T) * 8, "shift ammount out of bounds"); \
        return a << b; \
    }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_U_TYPE(T) \
    inline T op_lshr(T a, T b) { \
        anydsl_assert(b < sizeof(T) * 8, "shift ammount out of bounds"); \
        return a >> b; \
    }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_U_TYPE(T) \
    inline T op_ashr(T a, T b) { \
        anydsl_assert(b < sizeof(T) * 8, "shift ammount out of bounds"); \
        typedef make_signed< T >::type S; \
        return bcast< T , S>(bcast<S, T >(a) >> bcast<S, T >(b)); \
    }
#include "anydsl2/tables/primtypetable.h"

/*
 * arithops -- fp operations
 */

#define ANYDSL2_JUST_F_TYPE(T) inline T arithop_fadd(T a, T b) { return a + b; }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_F_TYPE(T) inline T arithop_fsub(T a, T b) { return a - b; }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_F_TYPE(T) inline T arithop_fmul(T a, T b) { return a * b; }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_F_TYPE(T) inline T arithop_fdiv(T a, T b) { return a / b; }
#include "anydsl2/tables/primtypetable.h"

#define ANYDSL2_JUST_F_TYPE(T) inline T arithop_frem(T a, T b) { return std::fmod(a, b); }
#include "anydsl2/tables/primtypetable.h"


// cmp_sgt
#define ANYDSL2_U_TYPE(T) \
    inline u1 op_cmp_sgt(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) > bcast<S, T>(b)); \
    }
#include "anydsl2/tables/primtypetable.h"

// cmp_sge
#define ANYDSL2_U_TYPE(T) \
    inline u1 op_cmp_sge(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) >= bcast<S, T>(b)); \
    }
#include "anydsl2/tables/primtypetable.h"

// cmp_slt
#define ANYDSL2_U_TYPE(T) \
    inline u1 op_cmp_slt(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) < bcast<S, T>(b)); \
    }
#include "anydsl2/tables/primtypetable.h"

// cmp_sle
#define ANYDSL2_U_TYPE(T) \
    inline u1 op_cmp_sle(T a, T b) { \
        typedef make_signed< T >::type S; \
        return u1(bcast<S, T>(a) <= bcast<S, T>(b)); \
    }
#include "anydsl2/tables/primtypetable.h"

// fcmp_oeq
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_oeq(T a, T b) { return u1(a == b); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_one
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_one(T a, T b) { return u1(a != b); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_ogt
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_ogt(T a, T b) { return u1(a > b); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_oge
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_oge(T a, T b) { return u1(a >= b); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_olt
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_olt(T a, T b) { return u1(a < b); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_ole
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_ole(T a, T b) { return u1(a <= b); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_ugt
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_ugt(T a, T b) { return u1(std::isgreater(a, b)); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_uge
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_uge(T a, T b) { return u1(std::isgreaterequal(a, b)); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_ult
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_ult(T a, T b) { return u1(std::isless(a, b)); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_ule
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_ule(T a, T b) { return u1(std::islessequal(a, b)); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_urd
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_urd(T a, T b) { return u1(std::isunordered(a, b)); }
#include "anydsl2/tables/primtypetable.h"

// fcmp_ord
#define ANYDSL2_F_TYPE(T) inline u1 op_fcmp_ord(T a, T b) { return u1(~fcmp_urd(a,b)); }
#include "anydsl2/tables/primtypetable.h"

#endif
#endif

} // namespace anydsl2
