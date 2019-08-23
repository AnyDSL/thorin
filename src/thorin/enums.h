#ifndef THORIN_ENUMS_H
#define THORIN_ENUMS_H

#include "thorin/util/types.h"

namespace thorin {

//------------------------------------------------------------------------------

enum ArithOpTag {
#define THORIN_ARITHOP(op) ArithOp_##op,
#include "thorin/tables/arithoptable.h"
};

enum CmpTag {
#define THORIN_CMP(op) Cmp_##op,
#include "thorin/tables/cmptable.h"
};

inline bool is_commutative(int tag) { return tag == ArithOp_add || tag == ArithOp_mul || tag == ArithOp_and || tag == ArithOp_or || tag == ArithOp_xor; }
inline bool is_associative(int tag) { return tag == ArithOp_add || tag == ArithOp_mul || tag == ArithOp_and || tag == ArithOp_or || tag == ArithOp_xor; }
CmpTag negate(CmpTag tag);

}

#endif
