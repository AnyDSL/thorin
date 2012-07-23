#include "anydsl/type.h"

#include "anydsl/literal.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, (IndexKind) kind, 0)
{
    debug = kind2str(this->kind());
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, IndexKind index, size_t num)
    : Type(world, index, num)
{}

CompoundType::CompoundType(World& world, IndexKind index, ArrayRef<const Type*> elems)
    : Type(world, index, elems.size())
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

} // namespace anydsl
