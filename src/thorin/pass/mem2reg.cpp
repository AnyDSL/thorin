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
        for (auto use : enter->out_frame()->uses()) {
            auto slot = use->as<Slot>();
            if (slot2info(slot).lattice == Lattice::SSA)
                set_val(slot, world().bot(slot->type()->pointee()));
        }
        man().new_state();
        return enter;
    } else if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == id()) {
        assert(analyze->num_ops() == 2);
        auto old_lam = analyze->op(0)->as_nominal<Lam>();
        auto slot    = analyze->op(1)->as<Slot>();
        if (auto new_lam = lam2info(old_lam).new_lam)
            return get_val(new_lam, slot);
    } else if (auto load = def->isa<Load>()) {
        if (auto slot = load->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return load;
            return world().tuple({load->mem(), get_val(slot)});
        }
    } else if (auto store = def->isa<Store>()) {
        if (auto slot = store->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return store;
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

        if (info.lattice == Lattice::SSA && old_lam->mem_param())
            man().new_state();

        if (auto& slots = info.slots; !slots.empty()) {
            Array<const Def*> types(slots.size(), [&](auto i) { return slots[i]->type()->pointee(); });
            auto new_domain = merge_sigma(old_lam->domain(), types);
            auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
            new2old(new_lam) = old_lam;
            info.new_lam = new_lam;
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
                set_val(new_lam, slots[i], new_lam->param(new_lam->num_params() - n + i));
        }
    }
}

const Def* Mem2Reg::get_val(Lam* lam, const Slot* slot) {
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

    // if we see this guy again during analyze, we need a phi in lam
    if (same == nullptr) {
        auto res = world().analyze(slot->type()->pointee(), {lam, slot}, id());
        outf("get_val {} for {}: {}\n", lam, slot, res);
        return res;
    }
    outf("get_val {} for {}: {}\n", lam, slot, same);
    return same;
}

void Mem2Reg::set_val(Lam* lam, const Slot* slot, const Def* val) {
    outf("set_val {} for {}: {}\n", lam, slot, val);
    lam2info(lam).slot2val[slot] = val;
}

void Mem2Reg::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    // we need to install a phi in lam next time around
    if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == id()) {
        assert(analyze->num_ops() == 2);
        auto lam  = analyze->op(0)->as_nominal<Lam>();
        auto slot = analyze->op(1)->as<Slot>();
        auto& info = lam2info(lam);

        if (info.lattice == Lattice::SSA) {
            info.slots.emplace_back(slot);
            outf("A: {}\n", slot);
            man().undo(info.undo);
        } else {
            //auto& info = slot2info(slot);
            //info.lattice = Lattice::Keep;
            outf("B: {}\n", slot);
            //man().undo(info.undo);
        }
        return;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto lam = op->isa_nominal<Lam>()) {
            auto& info = lam2info(lam);
            info.preds.emplace(man().cur_lam());

            if (info.lattice == SSA && (!def->isa<App>() || i != 0)) {
                outf("keep: {}\n", lam);
                info.lattice = Keep;
                man().undo(info.undo);
            }
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
