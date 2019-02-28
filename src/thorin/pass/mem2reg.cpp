#include "thorin/pass/mem2reg.h"

#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        slot2info(slot); // init
        return slot;
    }

    if (auto load = def->isa<Load>()) {
        if (auto slot = load->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return load;
            return world().tuple({load->mem(), get_val(mgr().cur_lam(), slot)});
        }
    }

    if (auto store = def->isa<Store>()) {
        if (auto slot = store->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return store;
            set_val(mgr().cur_lam(), slot, store->val());
            return store->mem();
        }
    }

    return def;
}

const Def* Mem2Reg::get_val(Lam* lam, const Slot* slot) {
    outf("get_val {} for {}\n", lam, slot);
    auto& slot2val = *lam2slot2val(lam);
    if (auto val = slot2val.lookup(slot))
        return *val;

    auto bot = world().bot(slot->type()->as<PtrType>()->pointee());
    const Def* same = bot;
    for (auto pred : lam2preds(lam)) {
        auto def = get_val(pred, slot);
        if (is_bot(def)) continue;
        if (is_bot(same) && same != def) {
            same = nullptr; // defs from preds are different
            break;
        }
        same = def;
    }

    if (same == nullptr)
        outf("xxx param in {} for {}\n", lam, slot);

    return world().bot(slot->type()->as<PtrType>()->pointee());
}

void Mem2Reg::set_val(Lam* lam, const Slot* slot, const Def* val) {
    outf("set_val {} for {}: {}\n", lam, slot, val);
    lam2slot2val(lam)->emplace(slot, val);
}

void Mem2Reg::analyze(const Def* def) {
    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>()) {
            lam2preds(lam).emplace(mgr().cur_lam());
        } else if (auto slot = op->isa<Slot>()) {
            if (auto& inf = slot2info(slot); inf.lattice == SSA) {
                inf.lattice = Keep;
                //mgr().undo(inf.undo);
            }
        }
    }
}

}