#include "anydsl/jump.h"

#include "anydsl/world.h"
#include "anydsl/primop.h"

namespace anydsl {

//------------------------------------------------------------------------------

Jump::Jump(World& world, const Def* to, const Def* const* begin, const Def* const* end) 
    : Def(Index_Goto, world.noret(), std::distance(begin, end) + 1)
{ 
    setOp(0, to);

    const Def* const* i = begin;
    for (size_t x = 1; i != end; ++x, ++i)
        setOp(x, *i);
}

const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

std::vector<const Lambda*> Jump::succ() const {
    std::vector<const Lambda*> result;

    if (const Lambda* lambda = to()->isa<Lambda>()) {
        result.push_back(lambda);
    } else if (const Select* select = to()->isa<Select>()) {
        const Lambda* tlambda = select->tdef()->as<Lambda>();
        const Lambda* flambda = select->fdef()->as<Lambda>();
        result.push_back(tlambda);
        result.push_back(flambda);
    } else {
        anydsl_assert(to()->as<Param>(), "unknown jump target");
    }

    return result;
}

//------------------------------------------------------------------------------

} // namespace anydsl
