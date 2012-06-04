#include "anydsl/type.h"

#include "anydsl/literal.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, (IndexKind) kind)
{
    debug = kind2str(this->kind());
}

//------------------------------------------------------------------------------

const Type* CompoundType::get(PrimLit* c) const { 
    anydsl_assert(isInteger(lit2type(c->kind())), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
