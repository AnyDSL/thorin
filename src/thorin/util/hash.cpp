#include "hash.h"

namespace thorin {

static uint16_t g_hash_gid_counter = 0;

uint16_t fetch_gid() {
    return g_hash_gid_counter++;
}


}
