#include "thorin/pass/mem2reg.h"

#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

Def* Mem2Reg::rewrite(Def* def) {
    if (auto lam = def->isa<Lam>()) {
        if (!lam2info(lam).slots.empty()) {
        }
    }

    return def;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto enter = def->isa<Enter>()) {
        for (auto use : enter->out_frame()->uses()) {
            slot2info(use->as<Slot>());
            auto slot = use->as<Slot>();
            outf("slot: {}\n", slot);
        }

        man().new_state();
        return enter;
    }

    if (auto load = def->isa<Load>()) {
        if (auto slot = load->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return load;
            return world().tuple({load->mem(), get_val(man().cur_lam(), slot)});
        }
    }

    if (auto store = def->isa<Store>()) {
        if (auto slot = store->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return store;
            set_val(man().cur_lam(), slot, store->val());
            return store->mem();
        }
    }

    return def;
}

const Def* Mem2Reg::get_val(Lam* lam, const Slot* slot) {
    outf("get_val {} for {}\n", lam, slot);
    auto& info = lam2info(lam);
    if (auto val = info.slot2val.lookup(slot)) {
        outf("get_val {} for {}: {}\n", lam, slot, *val);
        return *val;
    }

    auto bot = world().bot(slot->type()->as<PtrType>()->pointee());
    const Def* same = bot;
    for (auto pred : info.preds) {
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
    lam2info(lam).slot2val.emplace(slot, val);
}

void Mem2Reg::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>()) {
            auto& info = lam2info(lam);
            info.preds.emplace(man().cur_lam());
            outf("{} -> {}\n", man().cur_lam(), lam);
        } else if (auto slot = op->isa<Slot>()) {
            if (auto& info = slot2info(slot); info.lattice == SSA) {
                outf("keep: {}\n", slot);
                info.lattice = Keep;
                //man().undo(info.undo);
            }
        }
    }
}

}
