#include "anydsl/type.h"

#include "anydsl/literal.h"

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

CompoundType::CompoundType(World& world, IndexKind index, const Type* const* begin, const Type* const* end)
    : Type(world, index, std::distance(begin, end))
{
    size_t x = 0;
    for (const Type* const* i = begin; i != end; ++i, ++x)
        setOp(x, *i);
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
