#include "thorin/analyses/bta.h"

#include <iostream>
#include "thorin/be/thorin.h"
#include "thorin/primop.h"
#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

LV const LV::Top(LV::Dynamic);
LV const LV::Bot(LV::Static);

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
    if (get(def).isTop())
        return;

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
    /* Soundly overapproximate memory operations. */
    if (primOp->isa<MemOp>()) {
        update(primOp, LV::Top);
    }

    /* Join all operands. */
    LV lv = LV::Bot;
    for (auto op : primOp->ops())
        lv = lv.join(get(op));

    if (update(primOp, lv)) {
        /* TODO something changed, add dependent DefNodes to worklist */
    }
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
