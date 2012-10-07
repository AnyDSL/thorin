#ifndef IMPALA_INIT_H
#define IMPALA_INIT_H

#include "anydsl/world.h"

#include "impala/type.h"

namespace impala {

void init();
void destroy();

struct Init {
    Init() { init(); }
    ~Init() { destroy(); }

    TypeTable types;
    anydsl2::World world;
};

} // namespace impala

#endif
