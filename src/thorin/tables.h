#ifndef THORIN_TABLES_H
#define THORIN_TABLES_H

#include "thorin/util/utility.h"

namespace thorin {

using node_t   = u16;
using tag_t    = u32;
using flags_t  = u32;
using fields_t = u64;
using nat_t    = u64;

#define THORIN_NODE(m)                                                                  \
    m(Universe, universe) m(Kind, kind)                                                 \
    m(Pi, pi)             m(Lam, lam)           m(App, app)                             \
    m(Sigma, sigma)       m(Tuple, tuple)       m(Extract, extract) m(Insert, insert)   \
    m(Arr, arr)           m(Pack, pack)         m(Succ, succ)                           \
    m(Union, union_)      m(Which, which)                                               \
    m(Case, case_)        m(Ptrn, ptrn)                                                 \
    m(Match, match)                                                                     \
    m(Bot, bot) m(Top, top)                                                             \
    m(CPS2DS, cps2ds) m(DS2CPS, ds2cps)                                                 \
    m(Analyze, analyze)                                                                 \
    m(Axiom, axiom)                                                                     \
    m(Lit, lit)                                                                         \
    m(Nat, nat)                                                                         \
    m(Param, param)                                                                     \
    m(Global, global)
#define THORIN_TAG(m)                                                                                   \
    m(Mem, mem) m(Int, int) m(SInt, sint) m(Real, real) m(Ptr, ptr)                                     \
    m(Shr, shr) m(WOp, wop) m(ZOp, zop) m(ROp, rop) m(ICmp, icmp) m(RCmp, rcmp) m(Conv, conv) m(PE, pe) \
    m(Bit, bit)                                                                                         \
    m(Bitcast, bitcast) m(LEA, lea) m(Sizeof, sizeof)                                                   \
    m(Alloc, alloc) m(Slot, slot) m(Load, load) m(Store, store)                                         \
    m(Grad, grad) m(TangentType, tangent_type)


namespace WMode {
enum : nat_t {
    none = 0,
    nsw  = 1 << 0,
    nuw  = 1 << 1,
};
}

namespace RMode {
enum RMode : nat_t {
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
    bot = fast,
    top = none,
};
}

/// Integer operations that neither take a @p WMode nor do produce a side effect - arithmetic or logical shift right.
#define THORIN_SHR(m) m(Shr, a) m(Shr, l)
/// Integer operations that might wrap and, hence, take @p WMode.
#define THORIN_W_OP(m) m(WOp, add) m(WOp, sub) m(WOp, mul) m(WOp, shl)
/// Integer operations that might produce a "division by zero" side effect.
#define THORIN_Z_OP(m) m(ZOp, sdiv) m(ZOp, udiv) m(ZOp, smod) m(ZOp, umod)
/// Floating point (real) operations that take @p RMode.
#define THORIN_R_OP(m) m(ROp, add) m(ROp, sub) m(ROp, mul) m(ROp, div) m(ROp, mod)
/// Conversions
#define THORIN_CONV(m) m(Conv, s2s) m(Conv, u2u) m(Conv, s2r) m(Conv, u2r) m(Conv, r2s) m(Conv, r2u) m(Conv, r2r)
/// Partial Evaluation related operations
#define THORIN_PE(m) m(PE, hlt) m(PE, known) m(PE, run)

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

/**
 * Table for all binary boolean operations.
 * See https://en.wikipedia.org/wiki/Truth_table#Binary_operations
 *                                   x o x o                                */
#define THORIN_B_OP(m)            /* B B A A -                              */ \
                    m(Bit,     f) /* o o o o - always false                 */ \
                    m(Bit,   nor) /* o o o x -                              */ \
                    m(Bit, nciff) /* o o x o - not converse implication     */ \
                    m(Bit,    na) /* o o x x - not first argument           */ \
                    m(Bit,  niff) /* o x o o - not implication              */ \
                    m(Bit,    nb) /* o x o x - not second argument          */ \
                    m(Bit,  _xor) /* o x x o -                              */ \
                    m(Bit,  nand) /* o x x x -                              */ \
                    m(Bit,  _and) /* x o o o -                              */ \
                    m(Bit,  nxor) /* x o o x -                              */ \
                    m(Bit,     b) /* x o x o - second argument              */ \
                    m(Bit,   iff) /* x o x x - implication (if and only if) */ \
                    m(Bit,     a) /* x x o o - first argment                */ \
                    m(Bit,  ciff) /* x x o x - converse implication         */ \
                    m(Bit,   _or) /* x x x o -                              */ \
                    m(Bit,     t) /* x x x x - always true                  */

namespace Node {
#define CODE(node, name) node,
enum : node_t { THORIN_NODE(CODE) Max };
#undef CODE
}

namespace Tag {
#define CODE(tag, name) tag,
enum : tag_t { THORIN_TAG(CODE) Max };
#undef CODE
}

#define CODE(T, o) o,
enum class Bit    : tag_t { THORIN_B_OP  (CODE) };
enum class Shr    : tag_t { THORIN_SHR   (CODE) };
enum class WOp    : tag_t { THORIN_W_OP  (CODE) };
enum class ZOp    : tag_t { THORIN_Z_OP  (CODE) };
enum class ROp    : tag_t { THORIN_R_OP  (CODE) };
enum class ICmp   : tag_t { THORIN_I_CMP (CODE) };
enum class RCmp   : tag_t { THORIN_R_CMP (CODE) };
enum class Conv   : tag_t { THORIN_CONV  (CODE) };
enum class PE     : tag_t { THORIN_PE    (CODE) };
#undef CODE

constexpr ICmp operator|(ICmp a, ICmp b) { return ICmp(flags_t(a) | flags_t(b)); }
constexpr ICmp operator&(ICmp a, ICmp b) { return ICmp(flags_t(a) & flags_t(b)); }
constexpr ICmp operator^(ICmp a, ICmp b) { return ICmp(flags_t(a) ^ flags_t(b)); }

constexpr RCmp operator|(RCmp a, RCmp b) { return RCmp(flags_t(a) | flags_t(b)); }
constexpr RCmp operator&(RCmp a, RCmp b) { return RCmp(flags_t(a) & flags_t(b)); }
constexpr RCmp operator^(RCmp a, RCmp b) { return RCmp(flags_t(a) ^ flags_t(b)); }

#define CODE(T, o) case T::o: return #T "_" #o;
constexpr const char* op2str(Bit  o) { switch (o) { THORIN_B_OP (CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(Shr  o) { switch (o) { THORIN_SHR  (CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(WOp  o) { switch (o) { THORIN_W_OP (CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(ZOp  o) { switch (o) { THORIN_Z_OP (CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(ROp  o) { switch (o) { THORIN_R_OP (CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(ICmp o) { switch (o) { THORIN_I_CMP(CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(RCmp o) { switch (o) { THORIN_R_CMP(CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(Conv o) { switch (o) { THORIN_CONV (CODE) default: THORIN_UNREACHABLE; } }
constexpr const char* op2str(PE   o) { switch (o) { THORIN_PE   (CODE) default: THORIN_UNREACHABLE; } }
#undef CODE

namespace AddrSpace {
    enum : nat_t {
        Generic  = 0,
        Global   = 1,
        Texture  = 2,
        Shared   = 3,
        Constant = 4,
    };
}

// This trick let's us count the number of elements in an enum class without tainting it with an extra "Num" field.
template<class T> constexpr auto Num = size_t(-1);

#define CODE(T, o) + 1_s
constexpr auto Num_Nodes = 0_s THORIN_NODE(CODE);
constexpr auto Num_Tags  = 0_s THORIN_TAG (CODE);
template<> constexpr auto Num<Bit > = 0_s THORIN_B_OP (CODE);
template<> constexpr auto Num<Shr > = 0_s THORIN_SHR  (CODE);
template<> constexpr auto Num<WOp > = 0_s THORIN_W_OP (CODE);
template<> constexpr auto Num<ZOp > = 0_s THORIN_Z_OP (CODE);
template<> constexpr auto Num<ROp > = 0_s THORIN_R_OP (CODE);
template<> constexpr auto Num<ICmp> = 0_s THORIN_I_CMP(CODE);
template<> constexpr auto Num<RCmp> = 0_s THORIN_R_CMP(CODE);
template<> constexpr auto Num<Conv> = 0_s THORIN_CONV (CODE);
template<> constexpr auto Num<PE  > = 0_s THORIN_PE   (CODE);
#undef CODE

template<tag_t tag> struct Tag2Enum_   { using type = tag_t; };
template<> struct Tag2Enum_<Tag::Shr > { using type = Shr;   };
template<> struct Tag2Enum_<Tag::WOp > { using type = WOp;   };
template<> struct Tag2Enum_<Tag::ZOp > { using type = ZOp;   };
template<> struct Tag2Enum_<Tag::ROp > { using type = ROp;   };
template<> struct Tag2Enum_<Tag::ICmp> { using type = ICmp;  };
template<> struct Tag2Enum_<Tag::RCmp> { using type = RCmp;  };
template<> struct Tag2Enum_<Tag::Conv> { using type = Conv;  };
template<> struct Tag2Enum_<Tag::PE  > { using type = PE;    };
template<tag_t tag> using Tag2Enum = typename Tag2Enum_<tag>::type;

}

#endif
