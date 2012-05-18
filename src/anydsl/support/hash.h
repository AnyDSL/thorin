#ifndef ANYDSL_SUPPORT_HASH_H
#define ANYDSL_SUPPORT_HASH_H

namespace anydsl {

/*
 * NOTE sizeof-based runtime checks 
 * will be easily optimized away by every reasonable compiler.
 */

/**
 * Casts to uin64_t and shifts alignment away.
 *
 * 16 byte alignment on malloc/new is assumed if sizeof(ptr) == 8;
 * hence the shift amount of 4 to the right.
 *
 * 8 byte alignment on malloc/new is assumed otherwise;
 * hence the shift amount of 3 to the right.
 */
inline uint64_t ptr2u64(const void* p) { 
    int shift = sizeof(uintptr_t) == 8 ? 4 : 3;
    return uint64_t(uintptr_t(p)) >> shift; 
}

/// The index \p i is placed in the upper 8 bits of the 64 bit wide hash region.
inline uint64_t idx2u64(const IndexKind i) { return uint64_t(i) << (64 - 6); }

inline uint64_t hash1(const IndexKind ix) { return idx2u64(ix) << 7*8; }
inline uint64_t hash2(const IndexKind ix, const void* p) { return hash1(ix) | ptr2u64(p); }
inline uint64_t hash2(const IndexKind ix, uint64_t ui) { return hash1(ix) ^ ui; }
inline uint64_t hash3(const IndexKind ix, const void* p1, const void* p2) {
    if (sizeof(uintptr_t) == 8)
        return (ptr2u64(p1) + ptr2u64(p2)) | idx2u64(ix);
    else
        return ptr2u64(p1) | ptr2u64(p2) | idx2u64(ix);
}

} // namespace anydsl

#endif // ANYDSL_SUPPORT_HASH_H
