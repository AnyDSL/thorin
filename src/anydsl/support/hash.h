#ifndef ANYDSL_SUPPORT_HASH_H
#define ANYDSL_SUPPORT_HASH_H

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

} // namespace anydsl

#endif // ANYDSL_SUPPORT_HASH_H
