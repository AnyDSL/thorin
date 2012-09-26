#include "anydsl/type.h"

#include "anydsl/literal.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

//------------------------------------------------------------------------------

const Ptr* Type::to_ptr() const { return world().ptr(this); }

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, kind, 0)
{
    debug = kind2str(primtype_kind());
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, int kind, size_t num)
    : Type(world, kind, num)
{}

CompoundType::CompoundType(World& world, int kind, ArrayRef<const Type*> elems)
    : Type(world, kind, elems.size())
{
    size_t x = 0;
    for_all (elem, elems)
        set_op(x++, elem);
}

//------------------------------------------------------------------------------

size_t Sigma::hash() const {
    return named_ ? boost::hash_value(this) : CompoundType::hash();
}

bool Sigma::equal(const Def* other) const {
    return named_ ? this == other : CompoundType::equal(other);
}

//------------------------------------------------------------------------------

template<bool first_order>
bool Pi::classify_order() const {
    for_all (elem, elems())
        if (first_order ^ (elem->isa<Pi>() == 0))
            return false;

    return true;
}

bool Pi::is_fo() const { return classify_order<true>(); }
bool Pi::is_ho() const { return classify_order<false>(); }

//------------------------------------------------------------------------------

} // namespace anydsl
