#include "thorin/transform/cogen/cogen.h"

#include "thorin/analyses/bta.h"

namespace thorin {

void CoGen::run(World &world) {
    BTA bta;
    bta.run(world);
}

}
