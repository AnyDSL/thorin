#include "anydsl/air/binop.h"

#include "anydsl/air/type.h"
#include "anydsl/air/world.h"
#include "anydsl/support/hash.h"

namespace anydsl {

//------------------------------------------------------------------------------

uint64_t BinOp::hash() const {
    return hashBinOp(index(), luse.def(), ruse.def());
}

//------------------------------------------------------------------------------

RelOp::RelOp(ArithOpKind arithOpKind, Def* ldef, Def* rdef)
    : BinOp((IndexKind) arithOpKind, ldef->world().type_u1(), ldef, rdef)
{}

//------------------------------------------------------------------------------

} // namespace anydsl
