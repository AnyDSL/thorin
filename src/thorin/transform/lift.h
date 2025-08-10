#include "thorin/world.h"

namespace thorin {

enum LiftMode {
    Lift2Cff,
    ClosureConversion,
    JoinTargets
};

void lift(Thorin&, LiftMode mode);

}
