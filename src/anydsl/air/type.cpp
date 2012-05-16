#include "anydsl/air/type.h"

#include "anydsl/air/constant.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind primTypeKind)
    : Type(world, (IndexKind) primTypeKind, primTypeKind2str(primTypeKind))
{}

//------------------------------------------------------------------------------

const Type* CompoundType::get(PrimConst* c) const { 
    anydsl_assert(isInteger(c->primTypeKind()), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
