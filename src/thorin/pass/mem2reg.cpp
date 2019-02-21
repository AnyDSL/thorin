#include "thorin/pass/mem2reg.h"

#include "thorin/transform/mangle.h"
#include "thorin/util/iterator.h"

namespace thorin {

Mem2Reg::Info& Mem2Reg::info(const Slot* slot) {
    for (auto&& state : reverse_range(states_)) {
        if (auto i = state.slot2info.find(slot); i != state.slot2info.end())
            return i->second;
    }

    return cur_state().slot2info.emplace(slot, Info(Lattice::Bottom, PassMgr::No_Undo)).first->second;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        return slot;
    }

    return def;
}

void Mem2Reg::analyze(const Def*) {
}

}
