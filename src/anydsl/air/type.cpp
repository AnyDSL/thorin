#include "anydsl/air/type.h"

#include "anydsl/air/literal.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind primTypeKind)
    : Type(world, (IndexKind) primTypeKind)
{
    debug = primTypeKind2str(primTypeKind);
}

//------------------------------------------------------------------------------

const Type* CompoundType::get(PrimLit* c) const { 
    anydsl_assert(isInteger(c->primTypeKind()), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
