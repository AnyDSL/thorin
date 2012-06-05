#include "anydsl/literal.h"

#include "anydsl/type.h"
#include "anydsl/world.h"

namespace anydsl {

//------------------------------------------------------------------------------

bool PrimLit::equal(const Value* other) const {
    if (!Value::equal(other))
        return false;

    return box() == other->as<PrimLit>()->box();
}

size_t PrimLit::hash() const {
    size_t seed = Value::hash();
    boost::hash_combine(seed, bcast<u64, Box>(box()));

    return seed;
}

//------------------------------------------------------------------------------

} // namespace anydsl
