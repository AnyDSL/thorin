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

Def* Mem2Reg::rewrite(Def* def) {
    if (auto lam = def->isa<Lam>()) {
        auto& info = lam2info(lam);

        if (info.lattice == Lattice::SSA && lam->mem_param())
            man().new_state();

        auto& slots = info.slots;
        if (!slots.empty()) {
            man().new_state();
            size_t n = slots.size();
            Array<const Def*> types(n, [&](auto i) { return slots[i]->type()->pointee(); });
            auto new_domain = merge_sigma(lam->domain(), types);
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
    if (auto enter = def->isa<Enter>()) {
        for (auto use : enter->out_frame()->uses()) {
            auto slot = use->as<Slot>();
            slot2info(slot); // init optimistic info
            set_val(man().cur_lam(), slot, world().bot(slot->type()->pointee()));
        }

        man().new_state();
        return enter;
    } else if (auto load = def->isa<Load>()) {
        if (auto slot = load->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return load;
            return world().tuple({load->mem(), get_val(man().cur_lam(), slot)});
        }
    } else if (auto store = def->isa<Store>()) {
        if (auto slot = store->ptr()->isa<Slot>()) {
            if (slot2info(slot).lattice == Lattice::Keep) return store;
            set_val(man().cur_lam(), slot, store->val());
            return store->mem();
        }
    } else if (auto app = def->isa<App>()) {
        if (auto analyze = app->callee()->isa<Analyze>(); analyze && analyze->index() == PassMan::Pass_Index) {
            const auto& info = lam2info(analyze->op(0)->as_nominal<Lam>());
            Array<const Def*> args(info.slots.size(), [&](auto i) { return get_val(man().cur_lam(), info.slots[i]); });
            auto a = world().app(analyze->op(1), merge_tuple(app->arg(), args));
            a->dump();
            return a;
        }
    } else if (auto param = def->isa<Param>()) {
        const auto& info = lam2info(param->lam());
        outf("asdf: {}\n", info.slots.size());
    }

    return def;
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
        auto lam  = analyze->op(0)->as_nominal<Lam>();
        auto& info = lam2info(lam);

        if (analyze->num_ops() == 2) {
            auto slot = analyze->op(1)->as<Slot>();
            if (info.lattice == Lattice::SSA) {
                info.slots.emplace_back(slot);
                outf("A: {}\n", slot);
                man().undo(info.undo);
            } else {
                auto& info = slot2info(slot);
                info.lattice = Lattice::Keep;
                outf("B: {}\n", slot);
                man().undo(info.undo);
            }
        } else {
            outf("keep: {}\n", lam);
            info.lattice = Keep;
            if (!info.slots.empty()) {
                outf("C: {}\n", lam);
                man().undo(info.undo);
            }
        }
        return;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto lam = op->isa_nominal<Lam>()) {
            auto& info = lam2info(lam);
            info.preds.emplace(man().cur_lam());
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
