#include "anydsl/air/enums.h"

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
