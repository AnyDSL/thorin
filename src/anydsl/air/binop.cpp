#include "anydsl/air/binop.h"

#include "anydsl/air/type.h"
#include "anydsl/air/world.h"
#include "anydsl/support/hash.h"

namespace anydsl {

//------------------------------------------------------------------------------

/*static*/ uint64_t BinOp::hash(IndexKind index, const Def* ldef, const Def* rdef) {
    return hash3(index, ldef, rdef);
}

//------------------------------------------------------------------------------

RelOp::RelOp(RelOpKind kind, Def* ldef, Def* rdef)
    : BinOp((IndexKind) kind, ldef->world().type_u1(), ldef, rdef)
{
    anydsl_assert(ldef->type() == rdef->type(), "type are not equal");
}

//------------------------------------------------------------------------------

} // namespace anydsl
