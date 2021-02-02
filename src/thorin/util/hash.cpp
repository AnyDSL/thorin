#include "thorin/util/hash.h"

#include "thorin/util/stream.h"

namespace thorin {

hash_t hash(const char* s) {
    hash_t seed = thorin::hash_begin();
    for (const char* p = s; *p != '\0'; ++p)
        seed = thorin::hash_combine(seed, *p);
    return seed;
}

void debug_hash() {
    errf("debug with: break {}:{}", __FILE__, __LINE__);
}

}
