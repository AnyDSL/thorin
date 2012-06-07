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

Sigma::Sigma(World& world, size_t num)
    : Type(world, Index_Sigma, num)
    , named_(true)
{}

Sigma::Sigma(World& world, const Type* const* begin, const Type* const* end)
    : Type(world, Index_Sigma, std::distance(begin, end))
    , named_(false)
{
    size_t x = 0;
    for (const Type* const* i = begin; i != end; ++i, ++x)
        setOp(x, *i);
}

const Type* Sigma::get(const PrimLit* c) const { 
    anydsl_assert(isInteger(lit2type(c->kind())), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

Pi::Pi(const Sigma* sigma)
    : Type(sigma->world(), Index_Pi, 1)
{
    anydsl_assert(!sigma->named(), "only unnamed sigma allowed with pi type");
    setOp(0, sigma);
}

const Sigma* Pi::sigma() const { 
    return ops_[0]->as<Sigma>();
}

//------------------------------------------------------------------------------

} // namespace anydsl
