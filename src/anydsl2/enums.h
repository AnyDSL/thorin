#ifndef ANYDSL2_ENUMS_H
#define ANYDSL2_ENUMS_H

#include "anydsl2/util/types.h"

namespace anydsl2 {

//------------------------------------------------------------------------------


enum NodeKind {
#define ANYDSL2_GLUE(pre, next)
#define ANYDSL2_AIR_NODE(node, abbr) Node_##node,
#define ANYDSL2_PRIMTYPE(T) Node_PrimType_##T,
#define ANYDSL2_ARITHOP(op) Node_##op,
#define ANYDSL2_RELOP(op) Node_##op,
#define ANYDSL2_CONVOP(op) Node_##op,
#include "anydsl2/tables/allnodes.h"
};

enum Markers {
#define ANYDSL2_GLUE(pre, next) \
    End_##pre, \
    Begin_##next = End_##pre, \
    zzz##Begin_##next = Begin_##next - 1,
#define ANYDSL2_AIR_NODE(node, abbr) zzzMarker_##node,
#define ANYDSL2_PRIMTYPE(T) zzzMarker_PrimType_##T,
#define ANYDSL2_ARITHOP(op) zzzMarker_##op,
#define ANYDSL2_RELOP(op) zzzMarker_##op,
#define ANYDSL2_CONVOP(op) zzzMarker_##op,
#include "anydsl2/tables/allnodes.h"
    End_ConvOp,
    Begin_Node = 0,
    End_AllNodes    = End_ConvOp,

    Begin_AllNodes  = Begin_Node,

    Begin_PrimType  = Begin_PrimType_u,
    End_PrimType    = End_PrimType_f,

    Num_AllNodes    = End_AllNodes   - Begin_AllNodes,
    Num_Nodes       = End_Node       - Begin_Node,

    Num_PrimTypes_u = End_PrimType_u - Begin_PrimType_u,
    Num_PrimTypes_f = End_PrimType_f - Begin_PrimType_f,

    Num_ArithOps    = End_ArithOp    - Begin_ArithOp,
    Num_RelOps      = End_RelOp      - Begin_RelOp,
    Num_ConvOps     = End_ConvOp     - Begin_ConvOp,

    Num_PrimTypes = Num_PrimTypes_u + Num_PrimTypes_f,
};

enum PrimTypeKind {
#define ANYDSL2_UF_TYPE(T) PrimType_##T = Node_PrimType_##T,
#include "anydsl2/tables/primtypetable.h"
};

enum ArithOpKind {
#define ANYDSL2_ARITHOP(op) ArithOp_##op = Node_##op,
#include "anydsl2/tables/arithoptable.h"
};

enum RelOpKind {
#define ANYDSL2_RELOP(op) RelOp_##op = Node_##op,
#include "anydsl2/tables/reloptable.h"
};

enum ConvOpKind {
#define ANYDSL2_CONVOP(op) ConvOp_##op = Node_##op,
#include "anydsl2/tables/convoptable.h"
};

inline bool is_int(int kind)     { return (int) Begin_PrimType_u <= kind && kind < (int) End_PrimType_u; }
inline bool is_float(int kind)   { return (int) Begin_PrimType_f <= kind && kind < (int) End_PrimType_f; }
inline bool is_corenode(int kind){ return (int) Begin_AllNodes <= kind && kind < (int) End_AllNodes; }
inline bool is_primtype(int kind){ return (int) Begin_PrimType <= kind && kind < (int) End_PrimType; }
inline bool is_arithop(int kind) { return (int) Begin_ArithOp <= kind && kind < (int) End_ArithOp; }
inline bool is_relop(int kind)   { return (int) Begin_RelOp   <= kind && kind < (int) End_RelOp; }
inline bool is_convop(int kind)  { return (int) Begin_ConvOp  <= kind && kind < (int) End_ConvOp; }

inline bool is_div(int kind) { return  kind == ArithOp_sdiv || kind == ArithOp_udiv || kind == ArithOp_fdiv; }
inline bool is_rem(int kind) { return  kind == ArithOp_srem || kind == ArithOp_urem || kind == ArithOp_frem; }
inline bool is_bitop(int kind) { return  kind == ArithOp_and || kind == ArithOp_or || kind == ArithOp_xor; }
inline bool is_shift(int kind) { return  kind == ArithOp_shl || kind == ArithOp_lshr || kind == ArithOp_ashr; }
inline bool is_div_or_rem(int kind) { return is_div(kind) || is_rem(kind); }
inline bool is_commutative(int kind) { return kind == ArithOp_add  || kind == ArithOp_mul 
                                           || kind == ArithOp_fadd || kind == ArithOp_fmul 
                                           || kind == ArithOp_and  || kind == ArithOp_or || kind == ArithOp_xor; }
inline bool is_associative(int kind) { return kind == ArithOp_add || kind == ArithOp_mul
                                           || kind == ArithOp_and || kind == ArithOp_or || kind == ArithOp_xor; }

template<PrimTypeKind kind> struct kind2type {};
#define ANYDSL2_U_TYPE(T) template<> struct kind2type<PrimType_##T> { typedef T type; };
#define ANYDSL2_F_TYPE(T) template<> struct kind2type<PrimType_##T> { typedef T type; };
#include "anydsl2/tables/primtypetable.h"

template<class T> struct type2kind {};
template<> struct type2kind<bool> { static const PrimTypeKind kind = PrimType_u1; };
#define ANYDSL2_U_TYPE(T) template<> struct type2kind<T> { static const PrimTypeKind kind = PrimType_##T; };
#define ANYDSL2_F_TYPE(T) template<> struct type2kind<T> { static const PrimTypeKind kind = PrimType_##T; };
#include "anydsl2/tables/primtypetable.h"

const char* kind2str(PrimTypeKind kind);
int num_bits(PrimTypeKind);

RelOpKind negate(RelOpKind kind);

} // namespace anydsl2

#endif
