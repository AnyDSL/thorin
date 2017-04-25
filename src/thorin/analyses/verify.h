#ifndef THORIN_ANALYSES_VERIFY_H
#define THORIN_ANALYSES_VERIFY_H

namespace thorin {

class Continuation;
class World;

void verify(World& world);

#ifndef NDEBUG
inline void debug_verify(World& world) { verify(world); }
#else
inline void debug_verify(World&) {}
#endif

}

#endif
