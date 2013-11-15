#ifndef IMPALA_INIT_H
#define IMPALA_INIT_H

#include "thorin/world.h"

#include "impala/type.h"

namespace thorin {
    class World;
}

namespace impala {

void init();
void destroy();

struct Init {
    Init() { init(); }
    ~Init() { destroy(); }

    thorin::World world;
    TypeTable typetable;
};

} // namespace impala

#endif
