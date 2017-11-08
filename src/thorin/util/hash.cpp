#include "thorin/util/hash.h"

namespace thorin {

uint64_t hash(const char* s) {
    uint64_t seed = thorin::hash_begin();
    for (const char* p = s; *p != '\0'; ++p)
        seed = thorin::hash_combine(seed, *p);
    return seed;
}

void debug_hash() {
    VLOG("debug with: break {}:{}", __FILE__, __LINE__);
}

}
