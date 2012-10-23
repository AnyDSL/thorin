#ifndef IMPALA_INIT_H
#define IMPALA_INIT_H

#include "impala/type.h"

namespace impala {

void init();
void destroy();

struct Init {
    Init() { init(); }
    ~Init() { destroy(); }

    World world;
};

} // namespace impala

#endif
