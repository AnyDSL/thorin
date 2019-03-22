#include "thorin/pass/mem2reg.h"

#include "thorin/util.h"
#include "thorin/util/log.h"

namespace thorin {

const Slot* Mem2Reg::is_ssa_slot(const Def* ptr) {
    if (auto slot = ptr->isa<Slot>(); slot && slot2info(slot).lattice == SlotInfo::SSA)
        return slot;
    return nullptr;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        slot2info(slot); // init;
        set_val(slot->enter(), slot, world().bot(slot->type()->pointee()));
        return slot;
    } else if (auto load = def->isa<Load>()) {
        if (auto slot = is_ssa_slot(load->ptr()))
            return world().tuple({load->mem(), get_val(load->mem(), slot)});
    } else if (auto store = def->isa<Store>()) {
        if (auto slot = is_ssa_slot(store->ptr())) {
            set_val(store->mem(), slot, store->val());
            return store->mem();
        }
    } else if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            const auto& info = lam2info(lam);
            if (auto new_lam = info.new_lam) {
                Array<const Def*> args(info.slots.size(), [&](auto i) { return get_val(app->arg(0), info.slots[i]); });
                return world().app(new_lam, merge_tuple(app->arg(), args));
            }
        }
    }

    return def;
}

void Mem2Reg::inspect(Def* def) {
    if (auto old_lam = def->isa<Lam>()) {
        auto& info = lam2info(old_lam);
        if (old_lam->is_external() || old_lam->intrinsic() != Intrinsic::None || old_lam->mem_param() == nullptr) {
            info.lattice = LamInfo::Keep_Lam;
        } else if (info.lattice != LamInfo::Keep_Lam) {
            man().new_state();

            if (info.lattice == LamInfo::PredsN && !info.slots.empty()) {
                assert(old_lam->mem_param());
                Array<const Def*> types(info.slots.size(), [&](auto i) { return info.slots[i]->type()->pointee(); });
                auto new_domain = merge_sigma(old_lam->domain(), types);
                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                outf("new_lam: {} -> {}\n", old_lam, new_lam);
                new2old(new_lam) = old_lam;
                lam2info(new_lam).lattice = LamInfo::PredsN;
                info.new_lam = new_lam;
            }
        }
    }
}

void Mem2Reg::enter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        outf("enter: {}\n", new_lam);
        if (auto old_lam = new2old(new_lam)) {
            outf("enter: {}/{}\n", old_lam, new_lam);
            auto& slots = lam2info(old_lam).slots;
            size_t n = new_lam->num_params() - slots.size();

            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
            man().map(old_lam->param(), new_param);
            new_lam->set(old_lam->ops());

            for (size_t i = 0, e = slots.size(); i != e; ++i)
                set_val(new_lam->param(), slots[i], new_lam->param(n + i));
        }
    }
}

const Def* Mem2Reg::get_val(const Def* mem, const Slot* slot) {
    if (auto val = mem2slot2val(mem).lookup(slot)) {
        outf("get_val {} for {}: {}\n", mem, slot, *val);
        return *val;
    }

    if (auto param = mem->isa<Param>()) {
        const auto& info = lam2info(param->lam());
        switch (info.lattice) {
            case LamInfo::Preds0: return world().bot(slot->type()->pointee());
            case LamInfo::Preds1: return get_val(info.pred->body()->as<App>()->arg(0), slot);
            default: {
                auto old_lam = original(param->lam());
                outf("virtual phi: {}/{} for {}\n", old_lam, param->lam(), slot);
                return set_val(param, slot, world().analyze(slot->type()->pointee(), {old_lam, slot}, id()));
            }
        }
    }

    return get_val(mem->op(0), slot);
}

const Def* Mem2Reg::set_val(const Def* mem, const Slot* slot, const Def* val) {
    outf("set_val {} for {}: {}\n", mem, slot, val);
    return mem2slot2val(mem)[slot] = val;
}

void Mem2Reg::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    // we need to install a phi in lam next time around
    if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == id()) {
        assert(analyze->num_ops() == 2);
        auto lam  = analyze->op(0)->as_nominal<Lam>();
        auto slot = analyze->op(1)->as<Slot>();
        auto& lam_info = lam2info(lam);
        auto& slot_info = slot2info(slot);
        auto& slots = lam_info.slots;

        if (lam_info.lattice == LamInfo::Keep_Lam) {
            slot_info.lattice = SlotInfo::Keep_Slot;
            outf("keep: {}\n", slot);
            //man().undo(std::min(slot_info.undo, lam_info.undo));
            if (auto i = std::find(slots.begin(), slots.end(), slot); i != slots.end()) {
                slots.erase(i);
                man().undo(lam_info.undo);
            }
        } else {
            assert(lam_info.lattice == LamInfo::PredsN);
            assertf(std::find(slots.begin(), slots.end(), slot) == slots.end(), "already added slot {} to {}", slot, lam);
            //assert(slot_info.undo <= lam_info.undo);
            slots.emplace_back(slot);
            man().undo(lam_info.undo);
        }
        return;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto slot = op->isa<Slot>()) {
            if (auto& info = slot2info(slot); info.lattice == SlotInfo::SSA) {
                outf("keep: {}\n", slot);
                info.lattice = SlotInfo::Keep_Slot;
                man().undo(info.undo);
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            auto& info = lam2info(original(lam));
            auto pred = man().cur_lam();

            switch (info.lattice) {
                case LamInfo::Preds0:
                    info.lattice = LamInfo::Preds1;
                    info.pred = pred;
                    assert(info.slots.empty());
                    break;
                case LamInfo::Preds1:
                    info.lattice = LamInfo::PredsN;
                    man().undo(info.undo);
                    break;
                default:
                    break;
            }

            if (info.lattice == LamInfo::PredsN && (!def->isa<App>() || i != 0)) {
                info.lattice = LamInfo::Keep_Lam;
                outf("keep: {}\n", lam);
                if (!info.slots.empty()) {
                    info.slots.clear();
                    man().undo(info.undo);
                }
            }
        }
    }
}

}
