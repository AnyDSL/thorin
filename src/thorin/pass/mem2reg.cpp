#include "thorin/pass/mem2reg.h"

#include "thorin/transform/mangle.h"
#include "thorin/util/iterator.h"

namespace thorin {

Mem2Reg::Info& Mem2Reg::info(const Slot* slot) {
    for (auto&& state : reverse_range(states_)) {
        if (auto i = state.slot2info.find(slot); i != state.slot2info.end())
            return i->second;
    }

    return cur_state().slot2info.emplace(slot, Info(Lattice::SSA, mgr().num_states())).first->second;
}

ArrayRef<Lam*> Mem2Reg::lam2preds(Lam* lam) {
    for (auto&& state : reverse_range(states_)) {
        if (auto i = state.lam2preds.find(lam); i != state.lam2preds.end())
            return i->second;
    }

    return ArrayRef<Lam*>();
}

GIDMap<const Slot*, const Def*>& Mem2Reg::lam2slot2val(Lam* lam) {
    for (auto&& state : reverse_range(states_)) {
        if (auto i = state.lam2slot2val.find(lam); i != state.lam2slot2val.end())
            return *i->second;
    }

    return *cur_state().lam2slot2val.emplace(lam, std::make_unique<GIDMap<const Slot*, const Def*>>()).first->second;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        info(slot);
        return slot;
    }

    if (auto load = def->isa<Load>()) {
        if (auto slot = load->ptr()->isa<Slot>()) {
            if (info(slot).lattice == Lattice::Keep) return load;
            return world().tuple({load->mem(), get_val(mgr().cur_lam(), slot)});
        }
    }

    if (auto store = def->isa<Store>()) {
        if (auto slot = store->ptr()->isa<Slot>()) {
            if (info(slot).lattice == Lattice::Keep) return store;
            set_val(mgr().cur_lam(), slot, store->val());
            return store->mem();
        }
    }

    return def;
}

const Def* Mem2Reg::get_val(Lam* lam, const Slot* slot) {
    auto&& slot2val = lam2slot2val(lam);
    if (auto val = slot2val.lookup(slot))
        return *val;
    return world().bot(slot->type()->as<PtrType>()->pointee());
}

void Mem2Reg::set_val(Lam* lam, const Slot* slot, const Def* val) {
    lam2slot2val(lam).emplace(slot, val);
}

void Mem2Reg::analyze(const Def*) {
}

}
