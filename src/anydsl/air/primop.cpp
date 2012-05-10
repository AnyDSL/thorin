#include "anydsl/air/primop.h"

#include <typeinfo>

namespace anydsl {

inline uint64_t hash_helper(const IndexKind index, const Use& luse, const Use& ruse) {
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
        return (((uintptr_t) luse.def()) >> 4)
            +  (((uintptr_t) ruse.def()) >> 4)
            +  (((uintptr_t)      index) << 8*7);
    else
        return (((uintptr_t) luse.def()) >> 3)
            |  (((uintptr_t) ruse.def()) << (8*4 - 6))
            |  (((uintptr_t)      index) << (8*8 - 6));
}

uint64_t ArithOp::hash() const {
    return hash_helper(indexKind(), luse_, ruse_);
}

bool PrimOp::compare(PrimOp* other) const {
    if (this->hash() != other->hash())
        return false;

    if (typeid(*this) != typeid(*other))
        return false;

    if (this->indexKind() != other->indexKind())
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

} // namespace anydsl
