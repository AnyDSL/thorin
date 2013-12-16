#include "thorin/enums.h"

namespace thorin {

#define THORIN_GLUE(pre, next)
#define THORIN_AIR_NODE(node, abbr) static_assert(Node_##node == (NodeKind) zzzMarker_##node,             "NodeKind value not equal zzzMarker");
#define THORIN_PRIMTYPE(T)          static_assert(Node_PrimType_##T == (NodeKind) zzzMarker_PrimType_##T, "NodeKind value not equal zzzMarker");
#define THORIN_ARITHOP(op)          static_assert(Node_##op == (NodeKind) zzzMarker_##op,                 "NodeKind value not equal zzzMarker");
#define THORIN_RELOP(op)            static_assert(Node_##op == (NodeKind) zzzMarker_##op,                 "NodeKind value not equal zzzMarker");
#define THORIN_CONVOP(op)           static_assert(Node_##op == (NodeKind) zzzMarker_##op,                 "NodeKind value not equal zzzMarker");
#include "thorin/tables/allnodes.h"

const char* kind2str(NodeKind kind) {
    switch (kind) {
#define THORIN_GLUE(pre, next)
#define THORIN_PRIMTYPE(T)         case Node_PrimType_##T: return #T;
#define THORIN_AIR_NODE(n, abbr)   case Node_##n: return #n;
#define THORIN_ARITHOP(n)          case Node_##n: return #n;
#define THORIN_RELOP(n)            case Node_##n: return #n;
#define THORIN_CONVOP(n)           case Node_##n: return #n;
#include "thorin/tables/allnodes.h"
                                    default: THORIN_UNREACHABLE;
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
    THORIN_UNREACHABLE;
}

RelOpKind negate(CmpKind kind) {
    switch (kind) {
        case Cmp_eq: return Cmp_ne;
        case Cmp_ne: return Cmp_eq;
        case Cmp_lt: return Cmp_ge;
        case Cmp_le: return Cmp_gt;
        case Cmp_gt: return Cmp_le;
        case Cmp_ge: return Cmp_lt;
    }
    THORIN_UNREACHABLE;
}

} // namespace thorin
