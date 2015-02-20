#include "thorin/analyses/bta.h"
#include "thorin/primop.h"
#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

LV LV::join(LV other) const {
    return LV(Type(type | other.type));
}

void bta(World& world) {
}

}
