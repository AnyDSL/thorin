#ifndef ANYDSL2_ANALYSES_VERIFY_H
#define ANYDSL2_ANALYSES_VERIFY_H

#include "anydsl2/lambda.h"

namespace anydsl2 {

class World;

void verify(World& world);
void verify_closedness(World& world);

#ifndef NDEBUG
inline void debug_verify(World& world) { verify(world); }
#else
inline void debug_verify(World&) {}
#endif

} // namespace anydsl2

#endif
