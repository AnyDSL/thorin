#include "anydsl/enums.h"

#include <boost/static_assert.hpp>

namespace anydsl2 {

#define ANYDSL_GLUE(pre, next)
#define ANYDSL_AIR_NODE(node) BOOST_STATIC_ASSERT(Node_##node == (NodeKind) zzzMarker_##node);
#define ANYDSL_PRIMTYPE(T) BOOST_STATIC_ASSERT(Node_PrimType_##T == (NodeKind) zzzMarker_PrimType_##T);
#define ANYDSL_ARITHOP(op) BOOST_STATIC_ASSERT(Node_##op == (NodeKind) zzzMarker_##op);
#define ANYDSL_RELOP(op) BOOST_STATIC_ASSERT(Node_##op == (NodeKind) zzzMarker_##op);
#define ANYDSL_CONVOP(op) BOOST_STATIC_ASSERT(Node_##op == (NodeKind) zzzMarker_##op);
#include "anydsl/tables/allnodes.h"

const char* kind2str(PrimTypeKind kind) {
    switch (kind) {
#define ANYDSL_UF_TYPE(T) case PrimType_##T: return #T;
#include "anydsl/tables/primtypetable.h"
        default: ANYDSL_UNREACHABLE;
    }
}

} // namespace anydsl2
