#include "anydsl2/enums.h"

namespace anydsl2 {

#define ANYDSL2_GLUE(pre, next)
#define ANYDSL2_AIR_NODE(node, abbr) static_assert(Node_##node == (NodeKind) zzzMarker_##node,             "NodeKind value not equal zzzMarker");
#define ANYDSL2_PRIMTYPE(T)          static_assert(Node_PrimType_##T == (NodeKind) zzzMarker_PrimType_##T, "NodeKind value not equal zzzMarker");
#define ANYDSL2_ARITHOP(op)          static_assert(Node_##op == (NodeKind) zzzMarker_##op,                 "NodeKind value not equal zzzMarker");
#define ANYDSL2_RELOP(op)            static_assert(Node_##op == (NodeKind) zzzMarker_##op,                 "NodeKind value not equal zzzMarker");
#define ANYDSL2_CONVOP(op)           static_assert(Node_##op == (NodeKind) zzzMarker_##op,                 "NodeKind value not equal zzzMarker");
#include "anydsl2/tables/allnodes.h"

const char* kind2str(NodeKind kind) {
    switch (kind) {
#define ANYDSL2_GLUE(pre, next)
#define ANYDSL2_PRIMTYPE(T)         case Node_PrimType_##T: return #T;
#define ANYDSL2_AIR_NODE(n, abbr)   case Node_##n: return #n;
#define ANYDSL2_ARITHOP(n)          case Node_##n: return #n;
#define ANYDSL2_RELOP(n)            case Node_##n: return #n;
#define ANYDSL2_CONVOP(n)           case Node_##n: return #n;
#include "anydsl2/tables/allnodes.h"
                                    default: ANYDSL2_UNREACHABLE;
    }
}

int num_bits(PrimTypeKind kind) {
    switch (kind) {
        case PrimType_u1:  return 1;
        case PrimType_u8:  return 8;
        case PrimType_u16: return 16;
        case PrimType_u32: return 32;
        case PrimType_u64: return 64;
        case PrimType_f32: return 32;
        case PrimType_f64: return 64;
    }
    ANYDSL2_UNREACHABLE;
}

RelOpKind negate(RelOpKind kind) {
    switch (kind) {
        case RelOp_cmp_eq:   return RelOp_cmp_ne;
        case RelOp_cmp_ne:   return RelOp_cmp_eq;
        case RelOp_cmp_ult:  return RelOp_cmp_uge;
        case RelOp_cmp_ule:  return RelOp_cmp_ugt;
        case RelOp_cmp_ugt:  return RelOp_cmp_ule;
        case RelOp_cmp_uge:  return RelOp_cmp_ult;
        case RelOp_cmp_slt:  return RelOp_cmp_sge;
        case RelOp_cmp_sle:  return RelOp_cmp_sgt;
        case RelOp_cmp_sgt:  return RelOp_cmp_sle;
        case RelOp_cmp_sge:  return RelOp_cmp_slt;
        case RelOp_fcmp_oeq: return RelOp_fcmp_une;
        case RelOp_fcmp_one: return RelOp_fcmp_ueq;
        case RelOp_fcmp_olt: return RelOp_fcmp_uge;
        case RelOp_fcmp_ole: return RelOp_fcmp_ugt;
        case RelOp_fcmp_ogt: return RelOp_fcmp_ule;
        case RelOp_fcmp_oge: return RelOp_fcmp_ult;
        case RelOp_fcmp_ueq: return RelOp_fcmp_one;
        case RelOp_fcmp_une: return RelOp_fcmp_oeq;
        case RelOp_fcmp_ult: return RelOp_fcmp_oge;
        case RelOp_fcmp_ule: return RelOp_fcmp_ogt;
        case RelOp_fcmp_ugt: return RelOp_fcmp_ole;
        case RelOp_fcmp_uge: return RelOp_fcmp_olt;
        case RelOp_fcmp_uno: return RelOp_fcmp_ord;
        case RelOp_fcmp_ord: return RelOp_fcmp_uno;
    }
    ANYDSL2_UNREACHABLE;
}

} // namespace anydsl2
