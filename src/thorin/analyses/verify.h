#ifndef THORIN_ANALYSES_VERIFY_H
#define THORIN_ANALYSES_VERIFY_H

#include "thorin/lambda.h"

namespace thorin {

class World;

void verify(World& world);
/// \attention { You may only invoke this on a consistent world where all ops of all primops are \em not proxies. }
void verify_closedness(World& world);
void verify_calls(World& world);

#ifndef NDEBUG
inline void debug_verify(World& world) { verify(world); }
#else
inline void debug_verify(World&) {}
#endif

} // namespace thorin

#endif
