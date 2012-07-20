#include "anydsl/enums.h"

#include <boost/static_assert.hpp>

namespace anydsl {

#define ANYDSL_GLUE(pre, next)
#define ANYDSL_AIR_NODE(node) BOOST_STATIC_ASSERT(Index_##node == (IndexKind) zzzMarker_##node);
#define ANYDSL_PRIMTYPE(T) BOOST_STATIC_ASSERT(Index_PrimType_##T == (IndexKind) zzzMarker_PrimType_##T);
#define ANYDSL_PRIMLIT(T)  BOOST_STATIC_ASSERT(Index_PrimLit_##T == (IndexKind) zzzMarker_PrimLit_##T);
#define ANYDSL_ARITHOP(op) BOOST_STATIC_ASSERT(Index_##op == (IndexKind) zzzMarker_##op);
#define ANYDSL_RELOP(op) BOOST_STATIC_ASSERT(Index_##op == (IndexKind) zzzMarker_##op);
#define ANYDSL_CONVOP(op) BOOST_STATIC_ASSERT(Index_##op == (IndexKind) zzzMarker_##op);
#include "anydsl/tables/allindices.h"

const char* kind2str(PrimTypeKind kind) {
    switch (kind) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return #T;
#define ANYDSL_F_TYPE(T) case PrimType_##T: return #T;
#include "anydsl/tables/primtypetable.h"
        default: ANYDSL_UNREACHABLE;
    }
}

} // namespace anydsl
