#include "anydsl/air/literal.h"
#include "anydsl/air/type.h"
#include "anydsl/air/world.h"

#include "anydsl/util/foreach.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimLit::PrimLit(World& world, PrimTypeKind kind, Box box)
    : Literal((IndexKind) kind, world.type(kind))
    , box_(box)
{}

uint64_t PrimLit::hash() const {
    anydsl_assert(sizeof(Box) == 8, "Box has unexpected size");
    return (uint64_t(index()) << 32) | bcast<uint64_t, Box>((box()));
}

//------------------------------------------------------------------------------


uint64_t Tuple::hash() const {
    // TODO
    return 0;
}

//------------------------------------------------------------------------------

} // namespace anydsl
