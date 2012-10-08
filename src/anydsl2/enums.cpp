#include "anydsl2/enums.h"

#include <boost/static_assert.hpp>

namespace anydsl2 {

#define ANYDSL2_GLUE(pre, next)
#define ANYDSL2_AIR_NODE(node) BOOST_STATIC_ASSERT(Node_##node == (NodeKind) zzzMarker_##node);
#define ANYDSL2_PRIMTYPE(T) BOOST_STATIC_ASSERT(Node_PrimType_##T == (NodeKind) zzzMarker_PrimType_##T);
#define ANYDSL2_ARITHOP(op) BOOST_STATIC_ASSERT(Node_##op == (NodeKind) zzzMarker_##op);
#define ANYDSL2_RELOP(op) BOOST_STATIC_ASSERT(Node_##op == (NodeKind) zzzMarker_##op);
#define ANYDSL2_CONVOP(op) BOOST_STATIC_ASSERT(Node_##op == (NodeKind) zzzMarker_##op);
#include "anydsl2/tables/allnodes.h"

const char* kind2str(PrimTypeKind kind) {
    switch (kind) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return #T;
#include "anydsl2/tables/primtypetable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

} // namespace anydsl2
