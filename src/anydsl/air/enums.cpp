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

} // namespace anydsl
