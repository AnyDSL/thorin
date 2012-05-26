#include "anydsl/jump.h"

#include "anydsl/world.h"

namespace anydsl {

Jump::Jump(Lambda* from, Def* to)
    : Def(Index_Jump, to->world().noret(from->pi()))
    , from(*ops_append(from))
    , to(*ops_append(to))
{}

const NoRet* Jump::noret() const { 
    return type()->as<NoRet>(); 
}

} // namespace anydsl
