#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

Jump::Jump(Def* to, Def* const* begin, Def* const* end) 
    : Value(Index_Jump, 
            to->world().noret(to->type()->as<Pi>()), 
            std::distance(begin, end) + 1)
{ 
    setOp(0, to);

    Def* const* i = begin;
    for (size_t x = 1; i != end; ++x, ++i)
        setOp(x, *i);

#ifndef NDEBUG
    const Pi* pi = toLambda()->pi();
    anydsl_assert(pi->types().size() == args().size(), "element size of args and pi-to-type does not match");

    Types::const_iterator t = pi->types().begin();
    for_all (const& arg, args()) {
        anydsl_assert(arg.type() == *t, "type mismatch");
        ++t;
    }
#endif
}

//------------------------------------------------------------------------------


const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
