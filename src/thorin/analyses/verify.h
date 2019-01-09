#ifndef THORIN_ANALYSES_VERIFY_H
#define THORIN_ANALYSES_VERIFY_H

#include "thorin/config.h"

namespace thorin {

class Lam;
class World;

void verify(World& world);

#if THORIN_ENABLE_CHECKS
inline void debug_verify(World& world) { verify(world); }
#else
inline void debug_verify(World&) {}
#endif

}

#endif
