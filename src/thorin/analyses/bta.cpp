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

    // TODO Map the Memory Def to TOP.  This transitively covers all MemOps
    // TODO initialize worklist

    while (not worklist.empty()) {
        auto const def = worklist.back();
        worklist.pop_back();
        visit(def);
    }
}

LV BTA::get(DefNode const *def) {
    /* implicitly invokes the default constructor if no entry is present */
    return LatticeValues[def];
}

/// Updates the analysis information by joining the value of key `def` with `lv`.
/// \return true if the information changed
bool BTA::update(DefNode const *def, LV const lv) {
    auto it = LatticeValues.find(def);
    if (LatticeValues.end() == it) {
        LatticeValues.emplace(def, lv);
        return true;
    }

    LV const Old = it->second;
    LV const New = Old.join(lv);
    LatticeValues[def] = New;
    return New != Old;
}

void BTA::propagate(DefNode const *def, LV const lv) {
    if (not update(def, lv))
        return; // nothing changed
    for (auto use : def->uses())
        worklist.push_back(use);
}

void BTA::visit(DefNode const *def) {
    if (get(def).isTop())
        return;

    if (auto select = def->isa<Select>())
        return visit(select);
    if (auto primOp = def->isa<PrimOp>())
        return visit(primOp);
    if (auto param = def->isa<Param>())
        return visit(param);
    if (auto lambda = def->isa<Lambda>())
        return visit(lambda);

    THORIN_UNREACHABLE;
}

void BTA::visit(Lambda const *lambda) {
    /* The Binding Type of a lambda is defined by the binding type of its TO. */
    auto const to = lambda->to();
    return propagate(lambda, get(to));
}

void BTA::visit(Param const *param) {
    LV lv;
    for (auto arg : param->peek())
        lv = lv.join(get(arg.def()));
    propagate(param, lv);
}

void BTA::visit(PrimOp const *primOp) {
    LV lv;
    for (auto op : primOp->ops())
        lv = lv.join(get(op));
    propagate(primOp, lv);
}

void BTA::visit(Select const *select) {
    auto const ops  = select->ops();
    auto const cond = ops[0];
    auto const lhs  = ops[1];
    auto const rhs  = ops[2];

    if (not update(select, get(cond)))
        return; // nothing changed

    /* Add all uses of the select.  This includes the lambda "owning" the select. */
    for (auto use : select->uses())
        worklist.push_back(use);
    /* Add the successors of this lambda. */
    worklist.push_back(lhs);
    worklist.push_back(rhs);
}

//------------------------------------------------------------------------------

void bta(World& world) {
    BTA bta;
    bta.run(world);
    // TODO export results
}

}
