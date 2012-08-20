#include "anydsl/literal.h"

#include "anydsl/type.h"
#include "anydsl/world.h"

namespace anydsl {

//------------------------------------------------------------------------------

bool Literal::equal(const Def* other) const {
    if (!PrimOp::equal(other))
        return false;

    // also consider type
    return this->type() == other->type();
}

size_t Literal::hash() const {
    size_t seed = PrimOp::hash();
    // also consider type
    boost::hash_combine(seed, type());

    return seed;
}


//------------------------------------------------------------------------------

bool PrimLit::equal(const Def* other) const {
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

} // namespace anydsl
