#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

Jump::Jump(const Def* to, const Def* const* begin, const Def* const* end) 
    : Value(Index_Jump, 
            to->world().noret(to->type()->as<Pi>()), 
            std::distance(begin, end) + 1)
{ 
    setOp(0, to);

    const Def* const* i = begin;
    for (size_t x = 1; i != end; ++x, ++i)
        setOp(x, *i);

#ifndef NDEBUG
    const Pi* pi = toLambda()->pi();
    anydsl_assert(pi->sigma()->ops().size() == args().size(), "element size of args and pi-to-type does not match");

    const Use* t = pi->ops().begin();
    for_all (const& arg, args()) {
        anydsl_assert(arg.type() == t->def(), "type mismatch");
        ++t;
    }
#endif
}

//------------------------------------------------------------------------------


const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
