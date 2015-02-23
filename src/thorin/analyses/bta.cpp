#include "thorin/analyses/bta.h"

#include <iostream>
#include "thorin/be/thorin.h"
#include "thorin/primop.h"
#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

namespace {
LV LV_STATIC  = LV(LV::Static);
LV LV_DYNAMIC = LV(LV::Dynamic);
}

/** Computes the join of this lattice value with another. */
LV LV::join(LV other) const {
    return LV(Type(type | other.type));
}

std::string to_string(LV const lv) {
    switch (lv.type) {
        case LV::Static:  return "Static";
        case LV::Dynamic: return "Dynamic";
        default:          THORIN_UNREACHABLE;
    }
}

std::ostream & operator<<(std::ostream &os, LV const lv) {
    return os << to_string(lv);
}

void LV::dump(std::ostream &os) const {
    os << *this << std::endl;
}

void LV::dump() const {
    dump(std::cerr);
}

void bta(World& world) {
}

}
