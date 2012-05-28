#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

Jump::Jump(const ValueNumber& vn)
    : Value(Index_Jump, ((Def*) vn.more[0])->world().noret(((Def*) vn.more[0])->type()->as<Pi>()))
    , to(*ops_append(((Def*) vn.more[0])))
{
    for (size_t i = 1, e = vn.size; i != e; ++i)
        ops_append((Def*) vn.more[i]);

#ifndef NDEBUG
    const Pi* pi = toLambda()->pi();
    anydsl_assert(pi->types().size() == args().size(), "element size of args and pi-to-type does not match");

    Types::const_iterator t = pi->types().begin();
    FOREACH(const& arg, args()) {
        anydsl_assert(arg.type() == *t, "type mismatch");
        ++t;
    }
#endif
}

const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
