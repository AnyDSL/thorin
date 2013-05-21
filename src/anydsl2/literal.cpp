#include "anydsl2/literal.h"

#include "anydsl2/world.h"

namespace anydsl2 {

PrimLit::PrimLit(World& world, PrimTypeKind kind, Box box, const std::string& name)
    : Literal((int) kind, world.type(kind), name)
    , box_(box)
{}

}
