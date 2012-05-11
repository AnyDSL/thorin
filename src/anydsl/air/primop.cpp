#include "anydsl/air/primop.h"

#include <typeinfo>

#include "anydsl/air/type.h"

namespace anydsl {

inline uint64_t hashBinOp(const IndexKind index, const void* p1, const void* p2) {
    /*
     * The first variant assumes 16 byte alignment on malloc; 
     * hence the shift ammount of 4 to the right.
     * the index is being placed in the uppeer 8 bits
     *
     * The second variant assumes 8 byte alignment on malloc;
     * hence the shift ammount of 3 to the right.
     * The first def pointer is placed in the lower region,
     * the second one in the higher region,
     * the index is being placed in the upper (remaining) 6 bits
     */

    // NOTE the check will be easily optimized away by every reasonable compiler
    if (sizeof(uintptr_t) == 8)
        return (((uintptr_t)    p1) >> 4)
            +  (((uintptr_t)    p2) >> 4)
            +  (((uintptr_t) index) << 8*7);
    else
        return (((uintptr_t)    p1) >> 3)
            |  (((uintptr_t)    p2) << (8*4 - 6))
            |  (((uintptr_t) index) << (8*8 - 6));
}

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

uint64_t ArithOp::hash() const {
    return hashBinOp(index(), luse_.def(), ruse_.def());
}

//------------------------------------------------------------------------------

Extract::Extract(Def* tuple, PrimConst* elem, 
               const std::string& tupleDebug,
               const std::string debug)
    : PrimOp(Index_Extract, 
             scast<Sigma>(tuple->type())->get(elem),
             debug)
    , tuple_(tuple, this, tupleDebug)
    , elem_(elem)
{}

uint64_t Extract::hash() const {
    return hashBinOp(index(), tuple_.def(), elem_);
}

//------------------------------------------------------------------------------

Insert::Insert(Def* tuple, Def* value, PrimConst* elem, 
               const std::string& tupleDebug, const std::string& valueDebug,
               const std::string debug)
    : PrimOp(Index_Insert, 
             tuple->type(),
             debug)
    , tuple_(tuple, this, tupleDebug)
    , value_(value, this, valueDebug)
    , elem_(elem)
{
    anydsl_assert(tuple->getAs<Sigma>(), "must be of Sigma type");
}

//------------------------------------------------------------------------------

} // namespace anydsl
