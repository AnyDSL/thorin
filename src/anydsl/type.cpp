#include "anydsl/type.h"

#include "anydsl/literal.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, kind, 0)
{
    debug = kind2str(this->primtype_kind());
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
        setOp(x++, elem);
}

//------------------------------------------------------------------------------

size_t Sigma::hash() const {
    if (named_)
        return boost::hash_value(this);
    else
        return CompoundType::hash();
}

bool Sigma::equal(const Def* other) const {
    if (named_)
        return this == other;
    else
        return CompoundType::equal(other);
}

//------------------------------------------------------------------------------

template<bool first_order>
bool Pi::classify_order() const {
    for_all (elem, elems())
        if (first_order ^ (elem->isa<Pi>() == 0))
            return false;

    return true;
}

bool Pi::is_first_order() const { return classify_order<true>(); }
bool Pi::is_higher_order() const { return classify_order<false>(); }

//------------------------------------------------------------------------------

} // namespace anydsl
