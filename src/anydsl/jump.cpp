#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

//------------------------------------------------------------------------------

Jump::Jump(World& world, IndexKind indexKind, size_t numOps)
    : Def(indexKind, world.noret(), numOps)
{}

const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

//------------------------------------------------------------------------------

Goto::Goto(World& world, const Def* to, const Def* const* begin, const Def* const* end) 
    : Jump(world, Index_Goto, std::distance(begin, end) + 1)
{ 
    setOp(0, to);

    const Def* const* i = begin;
    for (size_t x = 1; i != end; ++x, ++i)
        setOp(x, *i);
}

//------------------------------------------------------------------------------

Branch::Branch(World& world, const Def* cond, 
               const Def* tto, const Def* const* tbegin, const Def* const* tend,
               const Def* fto, const Def* const* fbegin, const Def* const* fend) 
    : Jump(world, Index_Branch, std::distance(tbegin, tend) + std::distance(fbegin, fend) + 3)
{
    size_t arg = 0;
    setOp(arg++, cond);

    setOp(arg++, tto);
    for (const Def* const* i = tbegin; i != tend; ++i)
        setOp(arg++, *i);

    findex_ = arg;

    setOp(arg++, fto);
    for (const Def* const* i = fbegin; i != fend; ++i)
        setOp(arg++, *i);

}

//------------------------------------------------------------------------------

} // namespace anydsl
