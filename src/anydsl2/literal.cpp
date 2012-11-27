#include "anydsl2/literal.h"

#include "anydsl2/type.h"
#include "anydsl2/world.h"

namespace anydsl2 {

size_t hash_def(const PrimLitTuple& tuple) {
    size_t seed = hash_kind_type_size(tuple, 0);
    boost::hash_combine(seed, bcast<u64, Box>(tuple.get<2>()));
    return seed;
}

bool equal_def(const PrimLitTuple& tuple, const Def* other) {
    return equal_kind_type_size(tuple, 0, other) && tuple.get<2>() == other->as<PrimLit>()->box();
}

} // namespace anydsl2
