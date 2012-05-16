#include "anydsl/air/type.h"

#include "anydsl/air/constant.h"

namespace anydsl {

const Type* CompoundType::get(PrimConst* c) const { 
    anydsl_assert(isInteger(c->primTypeKind()), "must be an integer constant");
    return get(c->box().get_u64()); 
}

} // namespace anydsl
