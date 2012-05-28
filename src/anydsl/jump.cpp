#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

Jump::Jump(const ValueNumber& vn)
    : Value(Index_Jump, ((Def*) vn.more[0])->world().noret(((Def*) vn.more[0])->type()->as<Pi>()))
    , to(*ops_append(((Def*) vn.more[0])))
{
    for (size_t i = 1, e = vn.size; i != e; ++i)
        ops_append((Def*) vn.more[i]);
}

const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
