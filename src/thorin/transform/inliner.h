#ifndef THORIN_TRANSFORM_INLINER_H
#define THORIN_TRANSFORM_INLINER_H

namespace thorin {

class World;

void force_inline(Scope& scope, int threshold);
void inliner(World& world);

}

#endif
