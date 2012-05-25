#include "anydsl/air/jump.h"

#include "anydsl/air/world.h"

namespace anydsl {

Jump::Jump(Lambda* parent, Def* to)
    : Def(Index_Jump, to->world().noret(parent->pi()))
    , to(*ops_append(to))
{}

} // namespace anydsl
