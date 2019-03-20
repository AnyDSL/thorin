#include "thorin/pass/mem2reg.h"

#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

static Array<const Def*> merge(const Def* def, Defs defs) {
    return Array<const Def*>(defs.size() + 1, [&](auto i) { return i == 0 ? def : defs[i-1]; });
}

static Array<const Def*> merge_tuple_or_sigma(const Def* tuple_or_sigma, Defs defs) {
    Array<const Def*> result(tuple_or_sigma->num_ops() + defs.size());
    auto i = std::copy(tuple_or_sigma->ops().begin(), tuple_or_sigma->ops().end(), result.begin());
    std::copy(defs.begin(), defs.end(), i);
    return result;
}

static const Def* merge_sigma(const Def* def, Defs defs) {
    if (auto sigma = def->isa<Sigma>(); sigma && !sigma->isa_nominal())
        return def->world().sigma(merge_tuple_or_sigma(sigma, defs));
    return def->world().sigma(merge(def, defs));
}

static const Def* merge_tuple(const Def* def, Defs defs) {
    if (auto tuple = def->isa<Tuple>(); tuple && !tuple->type()->isa_nominal())
        return def->world().tuple(merge_tuple_or_sigma(tuple, defs));
    return def->world().tuple(merge(def, defs));
}

static Lam* find_mem_param(const Def* def) {
    if (auto param = def->isa<Param>()) return param->lam();
    return find_mem_param(def->op(0));
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        auto lam = find_mem_param(slot);
        assert(man().cur_state_id() != 0);
        slot2info(slot, SlotInfo(lam, man().cur_state_id())); // init
        return slot;
    //if (auto enter = def->isa<Enter>()) {
        //for (auto use : enter->out_frame()->uses()) {
            //auto lam = find_mem_param(enter);
            //assert(man().cur_state_id() != 0);
            //slot2info(use->as<Slot>(), SlotInfo(lam, man().cur_state_id())); // init
        //}
        //return enter;
    } else if (auto load = def->isa<Load>()) {
        if (auto slot = load->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == SlotInfo::Keep_Slot) return load;
            return world().tuple({load->mem(), get_val(slot)});
        }
    } else if (auto store = def->isa<Store>()) {
        if (auto slot = store->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == SlotInfo::Keep_Slot) return store;
            set_val(slot, store->val());
            return store->mem();
        }
    } else if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            const auto& info = lam2info(lam);
            if (auto new_lam = info.new_lam) {
                Array<const Def*> args(info.slots.size(), [&](auto i) { return get_val(info.slots[i]); });
                auto a = world().app(new_lam, merge_tuple(app->arg(), args));
                a->dump();
                return a;
            }
        }
    }

    return def;
}

void Mem2Reg::inspect(Def* def) {
    if (auto old_lam = def->isa<Lam>()) {
        auto& info = lam2info(old_lam);
        if (old_lam->is_external() || old_lam->intrinsic() != Intrinsic::None)
            info.lattice = LamInfo::Keep_Lam;
        if (info.lattice != LamInfo::Keep_Lam) {
            if (old_lam->mem_param()) man().new_state();

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
        lam2info(new_lam).slot2val.clear(); // remove any garbage from prevous runs
        outf("enter: {}\n", new_lam);
        if (auto old_lam = new2old(new_lam)) {
            outf("enter: {}/{}\n", old_lam, new_lam);
            auto& info = lam2info(old_lam);
            size_t n = new_lam->num_params() - info.slots.size();

            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
            man().map(old_lam->param(), new_param);
            new_lam->set(old_lam->ops());

            for (size_t i = 0, e = info.slots.size(); i != e; ++i)
                set_val(new_lam, info.slots[i], new_lam->param(n + i));
        }
    }
}

const Def* Mem2Reg::virtual_phi(Lam* new_lam, const Slot* slot) {
    auto old_lam = original(new_lam);
    outf("virtual phi: {}/{} for {}\n", old_lam, new_lam, slot);
    return set_val(new_lam, slot, world().analyze(slot->type()->pointee(), {old_lam, slot}, id()));
}

const Def* Mem2Reg::get_val(Lam* lam, const Slot* slot) {
    const auto& info = lam2info(lam);
    if (auto val = info.slot2val.lookup(slot)) {
        outf("get_val {} for {}: {}\n", lam, slot, *val);
        return *val;
    }

    if (slot2info(slot).lam == lam) return world().bot(slot->type()->pointee());

    switch (info.lattice) {
        case LamInfo::Preds0: return world().bot(slot->type()->pointee());
        case LamInfo::Preds1: return set_val(lam, slot, get_val(info.pred, slot));
        default: return virtual_phi(lam, slot);
    }
}

const Def* Mem2Reg::set_val(Lam* lam, const Slot* slot, const Def* val) {
    outf("set_val {} for {}: {}\n", lam, slot, val);
    return lam2info(lam).slot2val[slot] = val;
}

void Mem2Reg::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    // we need to install a phi in lam next time around
    if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == id()) {
        assert(analyze->num_ops() == 2);
        auto lam  = analyze->op(0)->as_nominal<Lam>();
        auto slot = analyze->op(1)->as<Slot>();
        auto& lam_info = lam2info(lam);

        if (lam_info.lattice == LamInfo::Keep_Lam) {
            auto& slot_info = slot2info(slot);
            slot_info.lattice = SlotInfo::Keep_Slot;
            outf("keep: {}\n", slot);
            man().undo(slot_info.undo);
        } else {
            auto& slots = lam_info.slots;
            assert(lam_info.lattice == LamInfo::PredsN);
            assertf(std::find(slots.begin(), slots.end(), slot) == slots.end(), "already added slot {} to {}", slot, lam);
            slots.emplace_back(slot);
            outf("A: {}\n", slot);
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
