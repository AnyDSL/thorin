#include "anydsl2/literal.h"

#include "anydsl2/type.h"
#include "anydsl2/world.h"

namespace anydsl2 {

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

u64 PrimLit::get_u64() const {
    switch (primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return box().get_##T();
#include "anydsl2/tables/primtypetable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2
