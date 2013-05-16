#include "anydsl2/literal.h"

#include "anydsl2/world.h"

namespace anydsl2 {

PrimLit::PrimLit(World& world, PrimTypeKind kind, Box box, const std::string& name)
    : Literal((int) kind, world.type(kind), name)
    , box_(box)
{}

Box PrimLit::value(size_t i) const {
    assert(i == 0);
    return box_;
}

size_t PrimLit::hash() const {
    size_t seed = Literal::hash();
    for (size_t i = 0, e = num_elems(); i != e; ++i)
        boost::hash_combine(seed, bcast<uint64_t, Box>(value(i)));
    return seed;
}

}
