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

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto enter = def->isa<Enter>()) {
        for (auto use : enter->out_frame()->uses())
            slot2info(use->as<Slot>());
        man().new_state();
        return enter;
    //} else if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == id()) {
        //assert(analyze->num_ops() == 2);
        //auto old_lam = analyze->op(0)->as_nominal<Lam>();
        //auto slot    = analyze->op(1)->as<Slot>();
        //if (auto new_lam = lam2info(old_lam).new_lam)
            //return get_val(new_lam, slot);
    } else if (auto load = def->isa<Load>()) {
        if (auto slot = load->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Keep) return load;
            return world().tuple({load->mem(), get_val(slot)});
        }
    } else if (auto store = def->isa<Store>()) {
        if (auto slot = store->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Keep) return store;
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
        if (old_lam->is_external() || old_lam->intrinsic() != Intrinsic::None) {
            info.lattice = Keep;
        } else if (info.lattice == SSA) {
            if (old_lam->mem_param()) man().new_state();

            if (info.lattice == SSA && !info.slots.empty()) {
                assert(old_lam->mem_param());
                Array<const Def*> types(info.slots.size(), [&](auto i) { return info.slots[i]->type()->pointee(); });
                auto new_domain = merge_sigma(old_lam->domain(), types);
                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                new2old(new_lam) = old_lam;
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
            man().map(old_lam->param(), new_lam->param(0));
            new_lam->set(old_lam->ops());

            auto& info = lam2info(old_lam);
            auto& slots = info.slots;
            size_t n = 1;

            for (size_t i = 0, e = slots.size(); i != e; ++i)
                set_val(new_lam, slots[i], new_lam->param(n + i));
        }
    }
}

const Def* Mem2Reg::virtual_phi(Lam* new_lam, const Slot* slot) {
    auto old_lam = new_lam;
    if (auto lam = new2old(new_lam)) old_lam = lam;
    outf("virtual phi: {}/{} for {}\n", old_lam, new_lam, slot);
    return set_val(new_lam, slot, world().analyze(slot->type()->pointee(), {old_lam, slot}, id()));
}

const Def* Mem2Reg::get_val(Lam* lam, const Slot* slot) {
    auto& info = lam2info(lam);

    if (auto val = info.slot2val.lookup(slot)) {
        outf("get_val {} for {}: {}\n", lam, slot, *val);
        return *val;
    }

    if (info.visited                                                                                                   // break cycle
            || std::any_of(info.preds.begin(), info.preds.end(), [&](auto pred) { return !man().has_entered(pred); })) // not sealed
        return virtual_phi(lam, slot);

    info.visited = true;
    const Def* same = world().bot(slot->type()->as<PtrType>()->pointee());
    for (auto pred : info.preds) {
        auto def = get_val(pred, slot);
        if (is_bot(def)) continue;

        if (is_bot(same)) {
            same = def;
        } else if (same != def) {
            same = nullptr; // defs from preds are different
            break;
        }
    }
    info.visited = false;

    if (same != nullptr) {
        outf("get_val {} for {}: {}\n", lam, slot, same);
        return same;
    }

    return virtual_phi(lam, slot);
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
        auto& info = lam2info(lam);

        if (info.lattice == SSA) {
            assertf(std::find(info.slots.begin(), info.slots.end(), slot) == info.slots.end(), "already added slot {} to {}", slot, lam);
            info.slots.emplace_back(slot);
            outf("A: {}\n", slot);
            man().undo(info.undo);
        } else {
            auto& info = slot2info(slot);
            info.lattice = Keep;
            outf("keep: {}\n", slot);
            man().undo(info.undo);
        }
        return;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto slot = op->isa<Slot>()) {
            if (auto& info = slot2info(slot); info.lattice == SSA) {
                outf("keep: {}\n", slot);
                info.lattice = Keep;
                man().undo(info.undo);
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            auto& info = lam2info(lam);

            auto pred = man().cur_lam();
            if (auto old_lam = new2old(lam))
                pred = old_lam;

            auto new_pred = info.preds.emplace(pred).second;

            if (info.lattice == SSA) {
                if (def->isa<App>() && i == 0) {
                    if (new_pred && man().has_entered(lam)) {
                        outf("discovered new pred: {} -> {}\n", pred, lam);
                        man().undo(info.undo);
                    }
                } else {
                    outf("keep: {}\n", lam);
                    info.lattice = Keep;
                    if (!info.slots.empty())
                        man().undo(info.undo);
                }
            }
        }
    }
}

}
