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

NoRet::NoRet(World& world, const Pi* pi)
    : Type(world, Index_NoRet, 1)
{
    setOp(0, pi);
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

const Type* CompoundType::get(const PrimLit* c) const { 
    anydsl_assert(isInteger(lit2type(c->kind())), "must be an integer constant");
    return get(c->box().get_u64()); 
}

} // namespace anydsl
