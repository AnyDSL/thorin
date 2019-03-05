#include "thorin/pass/mem2reg.h"

#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

static const Def* merge(const Def* a, Defs defs) {
    if (auto sigma = a->isa<Sigma>(); sigma && !sigma->isa_nominal()) {
        Array<const Def*> types(sigma->num_ops() + defs.size());
        auto i = std::copy(sigma->ops().begin(), sigma->ops().end(), types.begin());
        std::copy(defs.begin(), defs.end(), i);
        return a->world().sigma(types);
    }

    Array<const Def*> types(defs.size() + 1, [&](auto i) { return i == 0 ? a : defs[i-1]; });
    return a->world().sigma(types);
}

Def* Mem2Reg::rewrite(Def* def) {
    if (auto lam = def->isa<Lam>()) {
        auto& slots = lam2info(lam).slots;
        man().new_state();
        if (!slots.empty()) {
            size_t n = slots.size();
            Array<const Def*> types(n, [&](auto i) { return slots[i]->type()->pointee(); });
            auto new_domain = merge(lam->domain(), types);
            auto new_lam = world().lam(world().pi(new_domain, lam->codomain()), lam->debug());

            for (size_t i = 0, e = slots.size(); i != e; ++i)
                set_val(new_lam, slots[i], new_lam->param(new_lam->num_params() - n + i));

            outf("xxx new_lam\n");
            new_lam->dump_head();
            return new_lam;
        }
    }

    return def;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            //auto& info = lam2info(lam);
        }
    }

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
        return world().top(slot->type()->pointee());
    return same;
}

void Mem2Reg::set_val(Lam* lam, const Slot* slot, const Def* val) {
    outf("set_val {} for {}: {}\n", lam, slot, val);
    lam2info(lam).slot2val.emplace(slot, val);
}

void Mem2Reg::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    if (auto analyze = def->isa<Analyze>()) {
        auto lam  = analyze->op(0)->as_nominal<Lam>();
        auto slot = analyze->op(1)->as<Slot>();
        auto& info = lam2info(lam);
        info.slots.emplace_back(slot);
        man().undo(info.undo);
        return;
    }

    for (auto op : def->ops()) {
        if (auto lam = op->isa_nominal<Lam>()) {
            auto& info = lam2info(lam);
            info.preds.emplace(man().cur_lam());
            outf("{} -> {}\n", man().cur_lam(), lam);
        } else if (auto slot = op->isa<Slot>()) {
            if (auto& info = slot2info(slot); info.lattice == SSA) {
                outf("keep: {}\n", slot);
                info.lattice = Keep;
                man().undo(info.undo);
            }
        }
    }
}

}
