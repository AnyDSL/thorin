#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

#if 0
Jump::Jump(const ValueNumber& vn)
    : Value(Index_Jump, ((Def*) vn.more[0])->world().noret(((Def*) vn.more[0])->type()->as<Pi>()),
            vn.size)
{
    setOp(1, (Def*) vn.more[0]);
    for (size_t i = 1, e = vn.size; i != e; ++i)
        setOp(i + 1, (Def*) vn.more[i]);

#if 0
#ifndef NDEBUG
    const Pi* pi = toLambda()->pi();
    anydsl_assert(pi->types().size() == args().size(), "element size of args and pi-to-type does not match");

    Types::const_iterator t = pi->types().begin();
    FOREACH(const& arg, args()) {
        anydsl_assert(arg.type() == *t, "type mismatch");
        ++t;
    }
#endif
#endif
}
#endif

const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
