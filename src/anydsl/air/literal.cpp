#include "anydsl/air/literal.h"

#include "anydsl/air/type.h"
#include "anydsl/air/world.h"
#include "anydsl/util/foreach.h"
#include "anydsl/support/hash.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimLit::PrimLit(World& world, const ValueNumber& vn)
    : Literal((IndexKind) vn.index, world.type(lit2type((PrimLitKind) vn.index)))
{
    if (sizeof(void*) == sizeof(uint64_t))
        box_ = bcast<Box, uintptr_t>(vn.op1);
    else {
        Split split = {vn.op1, vn.op2};
        box_ = bcast<Box, Split>(split);
    }
}

ValueNumber PrimLit::VN(PrimLitKind kind, Box box) {
    if (sizeof(void*) == sizeof(uint64_t))
        return ValueNumber((IndexKind) kind, bcast<const void*, Box>(box));
    else {
        Split split = bcast<Split, Box>(box);
        return ValueNumber((IndexKind) kind, (const void*) split.op1, (const void*) split.op2);
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl
