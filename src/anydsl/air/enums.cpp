#include "anydsl/air/enums.h"

#include <boost/static_assert.hpp>

#if 0
BOOST_STATIC_ASSERT_MSG(anydsl::Num_Indexes < 64,
        "hash magic assumes number of nodes to be representable in 6 bits");
#endif

namespace anydsl {

const char* kind2str(PrimTypeKind kind) {
    switch (kind) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return #T;
#define ANYDSL_F_TYPE(T) case PrimType_##T: return #T;
#include "anydsl/tables/primtypetable.h"
        default: ANYDSL_UNREACHABLE;
    }
}

bool isArithOp(IndexKind kind) {
    switch (kind) {
#define ANYDSL_ARITHOP(op) case Index_##op: return true;
#include "anydsl/tables/arithoptable.h"
        default: return false;
    }
}

bool isRelOp(IndexKind kind) {
    switch (kind) {
#define ANYDSL_RELOP(op) case Index_##op: return true;
#include "anydsl/tables/reloptable.h"
        default: return false;
    }
}

bool isConvOp(IndexKind kind) {
    switch (kind) {
#define ANYDSL_CONVOP(op) case Index_##op: return true;
#include "anydsl/tables/convoptable.h"
        default: return false;
    }
}

} // namespace anydsl
