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

    Scope::for_each(world, [&](Scope const & s){
            worklist.push_back(s.entry());
    });

    /* all global variables are dynamic by default */
    for (auto primop : world.primops()) {
        if (auto global = primop->isa<Global>()) {
            update(global, LV::Top);
            worklist.push_back(global);
        }
    }

    /* functions and arguments called from outside are dynamic */
    for(auto lambda : world.lambdas()) {
        if(lambda->is_external() || lambda->cc() == CC::Device)
            for(auto param : lambda->params()) {
                update(param, LV::Top);
                worklist.push_back(param);
            }
    }

    while (not worklist.empty()) {
        auto const def = worklist.back();
        worklist.pop_back();
        visit(def);
    }
}

LV BTA::get(DefNode const *def) {
    auto it = LatticeValues.find(def);
    if (LatticeValues.end() == it)
        return LV::Bot;
    return it->second;
}

/// Updates the analysis information by joining the value of key `def` with `lv`.
/// If the information changed, adds `def` to the worklist.
/// \return true if the information changed
bool BTA::update(DefNode const *def, LV const lv) {
    auto it = LatticeValues.find(def);
    if (LatticeValues.end() == it) {
        LatticeValues.emplace(def, lv);
        worklist.push_back(def);
        return true;
    }

    LV const Old = it->second;
    LV const New = Old.join(lv);

    if (New != Old) {
        LatticeValues[def] = New;
        worklist.push_back(def);
        return true;
    }

    return false;
}

void BTA::visit(DefNode const *def) {
    std::cout << "Visiting DefNode " << def->unique_name() << "\n";
    LV const lv = get(def);
    for (auto const use : def->uses()) {
        if (auto select = use->isa<Select>()) {
            if (use.index() == 0)
                update(select, lv);
        } else if (auto primOp = use->isa<PrimOp>()) {
            update(primOp, lv);
        } else if (auto lambda = use->isa<Lambda>()) {
            /* Add unvisited immediate successors to the worklist. */
            for (auto const succ : lambda->direct_succs())
                update(succ, LV::Bot);

            if (use.index() == 0) { // Def is the TO of the using lambda
                if (update(lambda, lv)) { // A lambda is as least as dynamic as its TO
                    if (get(lambda).isTop()) { // If a lambda is dynamic, so are its parameters
                        for (auto const param : lambda->params())
                            update(param, LV::Top);
                    }
                }
            } else { // Def is an arg
                for (auto const succ : lambda->direct_succs()) {
                    auto const param = succ->params()[use.index() - 1];
                    update(param, lv);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------

void bta(World& world) {
    BTA bta;
    bta.run(world);
    // TODO export results
}

}
