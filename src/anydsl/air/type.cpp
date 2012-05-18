#include "anydsl/air/type.h"

#include "anydsl/air/literal.h"
#include "anydsl/support/hash.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, (IndexKind) kind)
{
    debug = kind2str(kind);
}

/*static*/ uint64_t PrimType::hash(PrimTypeKind kind) {
    return hash1((IndexKind) kind);
}

//------------------------------------------------------------------------------

const Type* CompoundType::get(PrimLit* c) const { 
    anydsl_assert(isInteger(lit2type(c->kind())), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
