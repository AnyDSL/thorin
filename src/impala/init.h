#ifndef IMPALA_INIT_H
#define IMPALA_INIT_H

#include "anydsl2/world.h"

#include "impala/type.h"

namespace anydsl2 {
    class World;
}

namespace impala {

void init();
void destroy();

struct Init {
    Init() { init(); }
    ~Init() { destroy(); }

    anydsl2::World world;
    TypeTable typetable;
};

} // namespace impala

#endif
