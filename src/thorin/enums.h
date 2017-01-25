#ifndef THORIN_ENUMS_H
#define THORIN_ENUMS_H

#include "thorin/util/types.h"

namespace thorin {

//------------------------------------------------------------------------------


enum NodeTag {
#define THORIN_GLUE(pre, next)
#define THORIN_NODE(node, abbr) Node_##node,
#define THORIN_PRIMTYPE(T) Node_PrimType_##T,
#define THORIN_ARITHOP(op) Node_##op,
#define THORIN_CMP(op) Node_##op,
#include "thorin/tables/allnodes.h"
};

enum Markers {
#define THORIN_GLUE(pre, next) \
    End_##pre, \
    Begin_##next = End_##pre, \
    zzz##Begin_##next = Begin_##next - 1,
#define THORIN_NODE(node, abbr) zzzMarker_##node,
#define THORIN_PRIMTYPE(T) zzzMarker_PrimType_##T,
#define THORIN_ARITHOP(op) zzzMarker_##op,
#define THORIN_CMP(op) zzzMarker_##op,
#include "thorin/tables/allnodes.h"
    End_Cmp,
    Begin_Node = 0,
    End_AllNodes    = End_Cmp,

    Begin_AllNodes  = Begin_Node,

    Begin_PrimType  = Begin_PrimType_bool,
    End_PrimType    = End_PrimType_qf,

    Num_AllNodes    = End_AllNodes   - Begin_AllNodes,
    Num_Nodes       = End_Node       - Begin_Node,

    Num_ArithOps    = End_ArithOp    - Begin_ArithOp,
    Num_Cmps        = End_Cmp        - Begin_Cmp,

    Num_PrimTypes   = End_PrimType_qf - Begin_PrimType_bool,
};

enum PrimTypeTag {
#define THORIN_ALL_TYPE(T, M) PrimType_##T = Node_PrimType_##T,
#include "thorin/tables/primtypetable.h"
};

enum ArithOpTag {
#define THORIN_ARITHOP(op) ArithOp_##op = Node_##op,
#include "thorin/tables/arithoptable.h"
};

enum CmpTag {
#define THORIN_CMP(op) Cmp_##op = Node_##op,
#include "thorin/tables/cmptable.h"
};

inline bool is_type_ps(int tag) { return (int) Begin_PrimType_ps <= tag && tag < (int) End_PrimType_ps; }
inline bool is_type_pu(int tag) { return (int) Begin_PrimType_pu <= tag && tag < (int) End_PrimType_pu; }
inline bool is_type_qs(int tag) { return (int) Begin_PrimType_qs <= tag && tag < (int) End_PrimType_qs; }
inline bool is_type_qu(int tag) { return (int) Begin_PrimType_qu <= tag && tag < (int) End_PrimType_qu; }
inline bool is_type_pf(int tag) { return (int) Begin_PrimType_pf <= tag && tag < (int) End_PrimType_pf; }
inline bool is_type_qf(int tag) { return (int) Begin_PrimType_qf <= tag && tag < (int) End_PrimType_qf; }

inline bool is_type_q(int tag) { return is_type_qs(tag) || is_type_qu(tag) || is_type_qf(tag); }
inline bool is_type_p(int tag) { return is_type_ps(tag) || is_type_pu(tag) || is_type_pf(tag); }
inline bool is_type_s(int tag) { return is_type_ps(tag) || is_type_qs(tag); }
inline bool is_type_u(int tag) { return is_type_pu(tag) || is_type_qu(tag); }
inline bool is_type_i(int tag) { return is_type_s (tag) || is_type_u (tag); }
inline bool is_type_f(int tag) { return is_type_pf(tag) || is_type_qf(tag); }

inline bool is_primtype(int tag){ return (int) Begin_PrimType <= tag && tag < (int) End_PrimType; }
inline bool is_arithop(int tag) { return (int) Begin_ArithOp <= tag && tag < (int) End_ArithOp; }
inline bool is_cmp(int tag)     { return (int) Begin_Cmp   <= tag && tag < (int) End_Cmp; }

inline bool is_bitop(int tag) { return  tag == ArithOp_and || tag == ArithOp_or || tag == ArithOp_xor; }
inline bool is_shift(int tag) { return  tag == ArithOp_shl || tag == ArithOp_shr; }
inline bool is_div_or_rem(int tag) { return tag == ArithOp_div || tag == ArithOp_rem; }
inline bool is_commutative(int tag) { return tag == ArithOp_add  || tag == ArithOp_mul
                                           || tag == ArithOp_and  || tag == ArithOp_or || tag == ArithOp_xor; }
inline bool is_associative(int tag) { return tag == ArithOp_add || tag == ArithOp_mul
                                           || tag == ArithOp_and || tag == ArithOp_or || tag == ArithOp_xor; }

template<PrimTypeTag tag> struct tag2type {};
#define THORIN_ALL_TYPE(T, M) template<> struct tag2type<PrimType_##T> { typedef T type; };
#include "thorin/tables/primtypetable.h"

template<class T> struct type2tag {};
#define THORIN_ALL_TYPE(T, M) template<> struct type2tag<T> { static const PrimTypeTag tag = PrimType_##T; };
#include "thorin/tables/primtypetable.h"

const char* tag2str(NodeTag tag);
int num_bits(PrimTypeTag);

CmpTag negate(CmpTag tag);

} // namespace thorin

#endif
