#include "thorin/union.h"

#include "thorin/rewrite.h"

namespace thorin {

bool Ptrn::is_trivial() const {
    return matcher()->isa<Param>() && matcher()->as<Param>()->nominal() == this;
}

bool Ptrn::matches(const Def* arg) const {
    return rewrite(as_nominal(), arg, 0) == arg;
}

}
