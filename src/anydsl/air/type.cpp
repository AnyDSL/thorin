#include "anydsl/air/type.h"

#include "anydsl/air/literal.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, const ValueNumber& vn)
    : Type(world, vn.index)
{
    debug = kind2str(kind());
}

//------------------------------------------------------------------------------

const Type* CompoundType::get(PrimLit* c) const { 
    anydsl_assert(isInteger(lit2type(c->kind())), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
