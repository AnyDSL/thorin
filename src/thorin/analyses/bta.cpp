#include "thorin/analyses/bta.h"
#include "thorin/primop.h"
#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

namespace {
LV LV_STATIC  = LV(LV::Static);
LV LV_DYNAMIC = LV(LV::Dynamic);
}

LV LV::join(LV other) const {
    return LV(Type(type | other.type));
}

void bta(World& world) {
}

}
