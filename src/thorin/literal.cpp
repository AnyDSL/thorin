#include "thorin/literal.h"

#include "thorin/world.h"

namespace thorin {

PrimLit::PrimLit(World& world, PrimTypeKind kind, Box box, const std::string& name)
    : Literal((NodeKind) kind, world.type(kind), name)
    , box_(box)
{}

}
