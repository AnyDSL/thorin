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
LV LV::join(LV const other) const {
    return LV(Type(type | other.type));
}

std::string to_string(LV const lv) {
    switch (lv.type) {
        case LV::Static:  return "Static";
        case LV::Dynamic: return "Dynamic";
        default:          THORIN_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

void BTA::run(World &world) {
    LatticeValues.clear();
    worklist.clear();

    // TODO initialize worklist

    while (not worklist.empty()) {
        auto const def = worklist.back();
        worklist.pop_back();
        visit(def);
    }
}

void BTA::visit(DefNode const *def) {
    if (auto lambda = def->isa<Lambda>())
        return visit(lambda);
    if (auto param = def->isa<Param>())
        return visit(param);
    if (auto primOp = def->isa<PrimOp>())
        return visit(primOp);
    THORIN_UNREACHABLE;
}

void BTA::visit(Lambda const *lambda) {
    // TODO implement
}

void BTA::visit(Param const *param) {
    // TODO implement
}

void BTA::visit(PrimOp const *primOp) {
    // TODO implement
}

LV BTA::get(DefNode const *def) {
    /* implicitly invokes the default constructor if no entry is present */
    return LatticeValues[def];
}

bool BTA::update(DefNode const *def, LV const lv) {
    LV const Old = get(def);
    LV const New = Old.join(lv);
    LatticeValues[def] = New;
    return New != Old;
}

//------------------------------------------------------------------------------

void bta(World& world) {
    BTA bta;
    bta.run(world);
    // TODO export results
}

}
