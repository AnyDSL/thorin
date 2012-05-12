#include "anydsl/air/primop.h"

#include <typeinfo>

#include "anydsl/air/type.h"
#include "anydsl/support/hash.h"
#include "anydsl/support/universe.h"

namespace anydsl {

bool PrimOp::compare(PrimOp* other) const {
    if (this->hash() != other->hash())
        return false;

    if (typeid(*this) != typeid(*other))
        return false;

    if (this->index() != other->index())
        return false;

    if (const ArithOp* a = dcast<ArithOp>(this)) {
        const ArithOp* b = scast<ArithOp>(other);

        if (a->luse().def() != b->luse().def())
            return false;

        if (a->luse().def() != b->ruse().def())
            return false;

        return false;
    }

    ANYDSL_UNREACHABLE;
}

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

} // namespace anydsl
