#include "thorin/enums.h"

#include "thorin/util/utility.h"

namespace thorin {

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

}
