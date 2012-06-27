#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

Jump::Jump(World& world, IndexKind indexKind, size_t numOps)
    : Def(indexKind, world.noret(), numOps)
{}

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


const NoRet* Goto::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
