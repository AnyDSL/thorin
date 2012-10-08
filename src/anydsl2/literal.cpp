#include "anydsl2/literal.h"

#include "anydsl2/type.h"
#include "anydsl2/world.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

bool Literal::equal(const Node* other) const {
    return PrimOp::equal(other) && type() == other->as<Literal>()->type();
}

size_t Literal::hash() const {
    size_t seed = PrimOp::hash();
    boost::hash_combine(seed, type());
    return seed;
}

//------------------------------------------------------------------------------

bool PrimLit::equal(const Node* other) const {
    if (!Literal::equal(other))
        return false;

    return box() == other->as<PrimLit>()->box();
}

size_t PrimLit::hash() const {
    size_t seed = Literal::hash();
    boost::hash_combine(seed, bcast<u64, Box>(box()));

    return seed;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
