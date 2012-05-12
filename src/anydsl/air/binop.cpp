#include "anydsl/air/binop.h"

#include "anydsl/air/type.h"
#include "anydsl/support/hash.h"
#include "anydsl/support/universe.h"

namespace anydsl {

//------------------------------------------------------------------------------

uint64_t BinOp::hash() const {
    return hashBinOp(index(), luse_.def(), ruse_.def());
}

//------------------------------------------------------------------------------

RelOp::RelOp(ArithOpKind arithOpKind, 
             Def* ldef, Def* rdef, 
             const std::string& ldebug, const std::string& rdebug,
             const std::string& debug)
    : BinOp((IndexKind) arithOpKind, ldef->universe().get_u1(), 
            ldef, rdef, ldebug, rdebug, debug)
{}

//------------------------------------------------------------------------------

} // namespace anydsl
