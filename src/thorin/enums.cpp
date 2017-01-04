#include "thorin/enums.h"

namespace thorin {

#define THORIN_GLUE(pre, next)
#define THORIN_NODE(node, abbr) static_assert(Node_##node == (NodeTag) zzzMarker_##node,                 "NodeTag value not equal zzzMarker");
#define THORIN_PRIMTYPE(T)          static_assert(Node_PrimType_##T == (NodeTag) zzzMarker_PrimType_##T, "NodeTag value not equal zzzMarker");
#define THORIN_ARITHOP(op)          static_assert(Node_##op == (NodeTag) zzzMarker_##op,                 "NodeTag value not equal zzzMarker");
#define THORIN_CMP(op)              static_assert(Node_##op == (NodeTag) zzzMarker_##op,                 "NodeTag value not equal zzzMarker");
#include "thorin/tables/allnodes.h"

const char* tag2str(NodeTag tag) {
    switch (tag) {
#define THORIN_GLUE(pre, next)
#define THORIN_PRIMTYPE(T)         case Node_PrimType_##T: return #T;
#define THORIN_NODE(n, abbr)   case Node_##n: return #n;
#define THORIN_ARITHOP(n)          case Node_##n: return #n;
#define THORIN_CMP(n)            case Node_##n: return #n;
#include "thorin/tables/allnodes.h"
                                    default: THORIN_UNREACHABLE;
    }
}

int num_bits(PrimTypeTag tag) {
    switch (tag) {
        case PrimType_bool: return 1;
        case PrimType_ps8:  case PrimType_qs8:  case PrimType_pu8:  case PrimType_qu8:  return 8;
        case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: case PrimType_pf16: case PrimType_qf16: return 16;
        case PrimType_ps32: case PrimType_qs32: case PrimType_pu32: case PrimType_qu32: case PrimType_pf32: case PrimType_qf32: return 32;
        case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: case PrimType_pf64: case PrimType_qf64: return 64;
    }
    THORIN_UNREACHABLE;
}

CmpTag negate(CmpTag tag) {
    switch (tag) {
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
