#include "thorin/world.h"

namespace thorin {

enum LiftMode {
    Lift2Cff,
    ClosureConversion,
    JoinTargets
};

void closure_conversion(Thorin&, LiftMode mode);

}
