#include "anydsl/literal.h"

#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

//------------------------------------------------------------------------------

#if 0
PrimLit::PrimLit(const ValueNumber& vn)
    : Literal((IndexKind) vn.index, (const Type*) vn.op3)
{
    if (sizeof(void*) == sizeof(uint64_t))
        box_ = bcast<Box, uintptr_t>(vn.op1);
    else {
        Split split = {vn.op1, vn.op2};
        box_ = bcast<Box, Split>(split);
    }
}

ValueNumber PrimLit::VN(const Type* t, Box box) {
    const PrimType* p = t->as<PrimType>();
    PrimLitKind litKind = type2lit(p->kind());
    IndexKind   indexKind = (IndexKind) litKind;
    if (sizeof(void*) == sizeof(uint64_t))
        return ValueNumber(indexKind, bcast<uintptr_t, Box>(box), 0, (uintptr_t) p);
    else {
        Split split = bcast<Split, Box>(box);
        return ValueNumber(indexKind, split.op1, split.op2, (uintptr_t) p);
    }
}

#endif

//------------------------------------------------------------------------------

} // namespace anydsl
