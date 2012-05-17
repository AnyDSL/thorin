#include "anydsl/air/type.h"

#include "anydsl/air/literal.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, (IndexKind) kind)
{
    debug = kind2str(kind);
}

//------------------------------------------------------------------------------

const Type* CompoundType::get(PrimLit* c) const { 
    anydsl_assert(isInteger(c->kind()), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
