#include "anydsl/air/literal.h"

#include "anydsl/air/type.h"
#include "anydsl/air/world.h"
#include "anydsl/util/foreach.h"
#include "anydsl/support/hash.h"

namespace anydsl {

//------------------------------------------------------------------------------

uint64_t Undef::hash(const Type* type) {
    return hash2(Index_Undef, type);
}

//------------------------------------------------------------------------------

uint64_t ErrorLit::hash(const Type* type) {
    return hash2(Index_ErrorLit, type);
}

//------------------------------------------------------------------------------

PrimLit::PrimLit(World& world, PrimLitKind kind, Box box)
    : Literal((IndexKind) kind, world.type(lit2type(kind)))
    , box_(box)
{}

/*static*/ uint64_t PrimLit::hash(PrimLitKind kind, Box box) {
    anydsl_assert(sizeof(Box) == 8, "Box has unexpected size");
    return hash2((IndexKind) kind, bcast<uint64_t, Box>(box));
}

//------------------------------------------------------------------------------

} // namespace anydsl
