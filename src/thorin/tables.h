#ifndef THORIN_TABLES_H
#define THORIN_TABLES_H

#include "thorin/util/utility.h"

namespace thorin {

#define THORIN_NODE(m) m(KindArity, *A)      /* don't  */ \
                       m(KindMulti, *M)      /* change */ \
                       m(KindStar,  *)       /* this   */ \
                       m(Universe, universe) /* order  */ \
                       m(App, app)                        \
                       m(Axiom, axiom)                    \
                       m(Bot, bot)                        \
                       m(Extract, extract)                \
                       m(Insert, insert)                  \
                       m(Lam, lam)                        \
                       m(Lit, lit)                        \
                       m(Pack, pack)                      \
                       m(Pi, pi)                          \
                       m(Sigma, sigma)                    \
                       m(Top, top)                        \
                       m(Tuple, tuple)                    \
                       m(Variadic, variadic)              \
                       m(VariantType, variant_type)       \
                       m(Bool, bool)                      \
                       m(Nat, nat)                        \
                       m(Mem, mem)                        \
                       m(Ptr, ptr)                        \
                       m(Alloc, alloc)                    \
                       m(Load, load)                      \
                       m(Store, store)                    \
                       m(Select, select)                  \
                       m(SizeOf, size_of)                 \
                       m(Global, global)                  \
                       m(Slot, slot)                      \
                       m(ArithOp, arithop)                \
                       m(Cmp, cmp)                        \
                       m(Cast, cast)                      \
                       m(Bitcast, bitcast)                \
                       m(Variant, variant)                \
                       m(LEA, lea)                        \
                       m(Hlt, hlt)                        \
                       m(Known, known)                    \
                       m(Run, run)                        \
                       m(Assembly, asm)                   \
                       m(Param, param)                    \
                       m(Analyze, analyze)                \
                       m(Sint, sint)                      \
                       m(Uint, uint)                      \
                       m(Real, real)

enum class WMode : uint64_t {
    none = 0,
    nsw  = 1 << 0,
    nuw  = 1 << 1,
};

enum class RMode : uint64_t {
    none     = 0,
    nnan     = 1 << 0, ///< No NaNs - Allow optimizations to assume the arguments and result are not NaN. Such optimizations are required to retain defined behavior over NaNs, but the value of the result is undefined.
    ninf     = 1 << 1, ///< No Infs - Allow optimizations to assume the arguments and result are not +/-Inf. Such optimizations are required to retain defined behavior over +/-Inf, but the value of the result is undefined.
    nsz      = 1 << 2, ///< No Signed Zeros - Allow optimizations to treat the sign of a zero argument or result as insignificant.
    arcp     = 1 << 3, ///< Allow Reciprocal - Allow optimizations to use the reciprocal of an argument rather than perform division.
    contract = 1 << 4, ///< Allow floating-point contraction (e.g. fusing a multiply followed by an addition into a fused multiply-and-add).
    afn      = 1 << 5, ///< Approximate functions - Allow substitution of approximate calculations for functions (sin, log, sqrt, etc). See floating-point intrinsic definitions for places where this can apply to LLVM’s intrinsic math functions.
    reassoc  = 1 << 6, ///< Allow reassociation transformations for floating-point operations. This may dramatically change results in floating point.
    finite   = nnan | ninf,
    unsafe   = nsz | arcp | reassoc,
    fast = nnan | ninf | nsz | arcp | contract | afn | reassoc,
};

/// Integer operations that might wrap and, hence, take @p WMode.
#define THORIN_W_OP(m) m(WOp, add) m(WOp, sub) m(WOp, mul) m(WOp, shl)
/// Integer operations that might produce a "division by zero" side effect.
#define THORIN_Z_OP(m) m(ZOp, sdiv) m(ZOp, udiv) m(ZOp, smod) m(ZOp, umod)
/// Integer operations that neither take a @p WMode nor do produce a side effect.
#define THORIN_I_OP(m) m(IOp, ashr) m(IOp, lshr) m(IOp, iand) m(IOp, ior) m(IOp, ixor)
/// Rloating point (float) operations that take @p RMode.
#define THORIN_R_OP(m) m(ROp, fadd) m(ROp, fsub) m(ROp, fmul) m(ROp, fdiv) m(ROp, fmod)
/// All cast operations that cast from/to float/signed/unsigned.
#define THORIN_CAST(m) m(_Cast, f2f) m(_Cast, f2s) m(_Cast, f2u) m(_Cast, s2f) m(_Cast, s2s) m(_Cast, u2f) m(_Cast, u2u)

/**
 * The 5 relations are disjoint and are organized as follows:
@verbatim
               ----
               4321
           01234567
           ////////
           00001111
     y→    00110011
           01010101

   x 0/000 ELLLXXXX
   ↓ 1/001 GELLXXXX
     2/010 GGELXXXX
     3/011 GGGEXXXX
  -4/4/100 YYYYELLL
  -3/5/101 YYYYGELL
  -2/6/110 YYYYGGEL   X = plus, minus
  -1/7/111 YYYYGGGE   Y = minus, plus
@endverbatim
 * The more obscure combinations are prefixed with @c _.
 * The standard comparisons front ends want to use, don't have this prefix.
 */
#define THORIN_I_CMP(m)              /* X Y G L E                                                   */ \
                     m(ICmp,   _f)   /* o o o o o - always false                                    */ \
                     m(ICmp,    e)   /* o o o o x - equal                                           */ \
                     m(ICmp,   _l)   /* o o o x o - less (same sign)                                */ \
                     m(ICmp,  _le)   /* o o o x x - less or equal                                   */ \
                     m(ICmp,   _g)   /* o o x o o - greater (same sign)                             */ \
                     m(ICmp,  _ge)   /* o o x o x - greater or equal                                */ \
                     m(ICmp,  _gl)   /* o o x x o - greater or less                                 */ \
                     m(ICmp, _gle)   /* o o x x x - greater or less or equal == same sign           */ \
                     m(ICmp,   _y)   /* o x o o o - minus plus                                      */ \
                     m(ICmp,  _ye)   /* o x o o x - minus plus or equal                             */ \
                     m(ICmp,   sl)   /* o x o x o - signed less                                     */ \
                     m(ICmp,  sle)   /* o x o x x - signed less or equal                            */ \
                     m(ICmp,   ug)   /* o x x o o - unsigned greater                                */ \
                     m(ICmp,  uge)   /* o x x o x - unsigned greater or equal                       */ \
                     m(ICmp, _ygl)   /* o x x x o - minus plus or greater or less                   */ \
                     m(ICmp,  _nx)   /* o x x x x - not plus minus                                  */ \
                     m(ICmp,   _x)   /* x o o o o - plus minus                                      */ \
                     m(ICmp,  _xe)   /* x o o o x - plus minus or equal                             */ \
                     m(ICmp,   ul)   /* x o o x o - unsigned less                                   */ \
                     m(ICmp,  ule)   /* x o o x x - unsigned less or equal                          */ \
                     m(ICmp,   sg)   /* x o x o o - signed greater                                  */ \
                     m(ICmp,  sge)   /* x o x o x - signed greater or equal                         */ \
                     m(ICmp, _xgl)   /* x o x x o - greater or less or plus minus                   */ \
                     m(ICmp,  _ny)   /* x o x x x - not minus plus                                  */ \
                     m(ICmp,  _xy)   /* x x o o o - different sign                                  */ \
                     m(ICmp, _xye)   /* x x o o x - different sign or equal                         */ \
                     m(ICmp, _xyl)   /* x x o x o - signed or unsigned less                         */ \
                     m(ICmp,  _ng)   /* x x o x x - signed or unsigned less or equal == not greater */ \
                     m(ICmp, _xyg)   /* x x x o o - signed or unsigned greater                      */ \
                     m(ICmp,  _nl)   /* x x x o x - signed or unsigned greater or equal == not less */ \
                     m(ICmp,   ne)   /* x x x x o - not equal                                       */ \
                     m(ICmp,   _t)   /* x x x x x - always true                                     */

#define THORIN_R_CMP(m)           /* U G L E                                 */ \
                     m(RCmp,   f) /* o o o o - always false                  */ \
                     m(RCmp,   e) /* o o o x - ordered and equal             */ \
                     m(RCmp,   l) /* o o x o - ordered and less              */ \
                     m(RCmp,  le) /* o o x x - ordered and less or equal     */ \
                     m(RCmp,   g) /* o x o o - ordered and greater           */ \
                     m(RCmp,  ge) /* o x o x - ordered and greater or equal  */ \
                     m(RCmp,  ne) /* o x x o - ordered and not equal         */ \
                     m(RCmp,   o) /* o x x x - ordered (no NaNs)             */ \
                     m(RCmp,   u) /* x o o o - unordered (either NaNs)       */ \
                     m(RCmp,  ue) /* x o o x - unordered or equal            */ \
                     m(RCmp,  ul) /* x o x o - unordered or less             */ \
                     m(RCmp, ule) /* x o x x - unordered or less or equal    */ \
                     m(RCmp,  ug) /* x x o o - unordered or greater          */ \
                     m(RCmp, uge) /* x x o x - unordered or greater or equal */ \
                     m(RCmp, une) /* x x x o - unordered or not equal        */ \
                     m(RCmp,   t) /* x x x x - always true                   */

enum class WOp : u64 {
#define CODE(T, o) o,
    THORIN_W_OP(CODE)
#undef CODE
};

enum class ZOp : u64 {
#define CODE(T, o) o,
    THORIN_Z_OP(CODE)
#undef CODE
};

enum class IOp : u64 {
#define CODE(T, o) o,
    THORIN_I_OP(CODE)
#undef CODE
};

enum class ROp : u64 {
#define CODE(T, o) o,
    THORIN_R_OP(CODE)
#undef CODE
};

enum class ICmp : u64 {
#define CODE(T, o) o,
    THORIN_I_CMP(CODE)
#undef CODE
};

enum class RCmp : u64 {
#define CODE(T, o) o,
    THORIN_R_CMP(CODE)
#undef CODE
};

enum class _Cast : u64 {
#define CODE(T, o) o,
    THORIN_CAST(CODE)
#undef CODE
};

constexpr WMode operator|(WMode a, WMode b) { return WMode(int64_t(a) | int64_t(b)); }
constexpr WMode operator&(WMode a, WMode b) { return WMode(int64_t(a) & int64_t(b)); }

constexpr RMode operator|(RMode a, RMode b) { return RMode(int64_t(a) | int64_t(b)); }
constexpr RMode operator&(RMode a, RMode b) { return RMode(int64_t(a) & int64_t(b)); }

constexpr ICmp operator|(ICmp a, ICmp b) { return ICmp(int64_t(a) | int64_t(b)); }
constexpr ICmp operator&(ICmp a, ICmp b) { return ICmp(int64_t(a) & int64_t(b)); }

constexpr RCmp operator|(RCmp a, RCmp b) { return RCmp(int64_t(a) | int64_t(b)); }
constexpr RCmp operator&(RCmp a, RCmp b) { return RCmp(int64_t(a) & int64_t(b)); }

constexpr bool has_feature(WMode mode, WMode feature) { return (mode & feature) == feature; }
constexpr bool has_feature(RMode mode, RMode feature) { return (mode & feature) == feature; }

constexpr const char* op2str(WOp o) {
    switch (o) {
#define CODE(T, o) case T::o: return #o;
    THORIN_W_OP(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

constexpr const char* op2str(ZOp o) {
    switch (o) {
#define CODE(T, o) case T::o: return #o;
    THORIN_Z_OP(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

constexpr const char* op2str(IOp o) {
    switch (o) {
#define CODE(T, o) case T::o: return #o;
    THORIN_I_OP(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

constexpr const char* op2str(ROp o) {
    switch (o) {
#define CODE(T, o) case T::o: return #o;
    THORIN_R_OP(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

constexpr const char* op2str(ICmp o) {
    switch (o) {
#define CODE(T, o) case T::o: return "icmp_" #o;
    THORIN_I_CMP(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

constexpr const char* op2str(RCmp o) {
    switch (o) {
#define CODE(T, o) case T::o: return "rcmp_" #o;
    THORIN_R_CMP(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

constexpr const char* cast2str(_Cast o) {
    switch (o) {
#define CODE(T, o) case T::o: return #o;
    THORIN_CAST(CODE)
#undef CODE
        default: THORIN_UNREACHABLE;
    }
}

}

namespace thorin {

template<class T> constexpr auto Num = size_t(-1);

#define CODE(T, o) + 1_s
template<> constexpr auto Num<WOp>  = 0_s THORIN_W_OP (CODE);
template<> constexpr auto Num<ZOp>  = 0_s THORIN_Z_OP (CODE);
template<> constexpr auto Num<IOp>  = 0_s THORIN_I_OP (CODE);
template<> constexpr auto Num<ROp>  = 0_s THORIN_R_OP (CODE);
template<> constexpr auto Num<ICmp> = 0_s THORIN_I_CMP(CODE);
template<> constexpr auto Num<RCmp> = 0_s THORIN_R_CMP(CODE);
template<> constexpr auto Num<_Cast> = 0_s THORIN_CAST (CODE);
#undef CODE

}

#endif

