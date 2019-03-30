#include "thorin/pass/mem2reg.h"

#include "thorin/util.h"
#include "thorin/util/log.h"

namespace thorin {

static const Def* proxy_type(const Analyze* proxy) { return proxy->type()->as<PtrType>()->pointee(); }

const Analyze* Mem2Reg::isa_proxy(const Def* ptr) {
    if (auto analyze = ptr->isa<Analyze>(); analyze && analyze->index() == index() && !analyze->op(1)->isa<Analyze>()) return analyze;
    return nullptr;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        auto orig = original(man().cur_lam());
        auto& info = lam2info(orig);
        auto slot_id = info.num_slots++;
        auto proxy = world().analyze(slot->out_ptr_type(), {orig, world().lit_nat(slot_id)}, index(), slot->debug());
        auto& lattice = info.slots[slot_id];

        outf("slot: {}/{}\n", proxy, lattice);
        if (lattice == Info::SSA) {
            //man().new_state();
            set_val(proxy, world().bot(proxy_type(proxy)));
            auto& info = lam2info(man().cur_lam());
            info.writable.emplace(proxy);
            return world().tuple({slot->mem(), proxy});
        }
    } else if (auto load = def->isa<Load>()) {
        if (auto proxy = isa_proxy(load->ptr()))
            return world().tuple({load->mem(), get_val(proxy)});
    } else if (auto store = def->isa<Store>()) {
        if (auto proxy = isa_proxy(store->ptr())) {
            if (lam2info(man().cur_lam()).writable.contains(proxy)) {
                set_val(proxy, store->val());
                return store->mem();
            }
        }
    } else if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            const auto& info = lam2info(lam);
            if (auto new_lam = info.new_lam) {
                outf("{}/{}:\n", lam2info(lam).undo, lam2info(new_lam).undo);
                Array<const Def*> args(info.phis.size(), [&](auto i) { return get_val(info.phis[i]); });
                return world().app(new_lam, merge_tuple(app->arg(), args));
            }
        }
    }

    return def;
}

void Mem2Reg::inspect(Def* def) {
    if (auto old_lam = def->isa<Lam>()) {
        auto& info = lam2info(old_lam);
        if (old_lam->is_external() || old_lam->intrinsic() != Intrinsic::None) {
            info.lattice = Info::Keep;
        } else if (info.lattice != Info::Keep) {
            man().new_state();

            if (info.lattice == Info::PredsN && !info.phis.empty()) {

                /*
                for (size_t i = 0; i != info.phis.size();) {
                    if (proxy2info(info.phis[i]).lattice == Info::Keep_) {
                        info.phis.erase(info.phis.begin() + i);
                        outf("remove proxy: {}\n", info.phis[i]);
                    } else
                        ++i;
                }
                */

                assert(old_lam->mem_param());
                Array<const Def*> types(info.phis.size(), [&](auto i) { return proxy_type(info.phis[i]); });
                auto new_domain = merge_sigma(old_lam->domain(), types);
                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                outf("new_lam: {} -> {}\n", old_lam, new_lam);
                new2old(new_lam) = old_lam;
                lam2info(new_lam).lattice = Info::PredsN;
                info.new_lam = new_lam;
            }
        }
    }
}

void Mem2Reg::enter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        outf("enter: {}\n", new_lam);
        auto& info = lam2info(new_lam);
        info.proxy2val.clear();
        if (auto old_lam = new2old(new_lam)) {
            auto& phis = lam2info(old_lam).phis;

            outf("enter: {}/{}\n", old_lam, new_lam);
            size_t n = new_lam->num_params() - phis.size();

            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
            man().map(old_lam->param(), new_param);
            new_lam->set(old_lam->ops());

            if (auto old_lam = new2old(new_lam)) {
                outf("enter: {}/{}\n", old_lam, new_lam);
                auto& phis = lam2info(old_lam).phis;
                size_t n = new_lam->num_params() - phis.size();

                for (size_t i = 0, e = phis.size(); i != e; ++i) {
                    auto proxy = phis[i];
                    //auto lam = proxy->op(0)->as_nominal<Lam>();
                    set_val(new_lam, proxy, new_lam->param(n + i));
                }
            }
        }
    }
}

void Mem2Reg::reenter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        outf("reenter: {}\n", new_lam);
        // remove any potential garbage from previous runs
        auto& info = lam2info(new_lam);
        info.num_slots = 0;
        //if (auto old_lam = new2old(new_lam)) {
            //outf("enter: {}/{}\n", old_lam, new_lam);
            //auto& phis = lam2info(old_lam).phis;
            //size_t n = new_lam->num_params() - phis.size();

            //for (size_t i = 0, e = phis.size(); i != e; ++i) {
                //auto proxy = phis[i];
                ////auto lam = proxy->op(0)->as_nominal<Lam>();
                //set_val(new_lam, proxy, new_lam->param(n + i));
            //}
        //}
    }
}

const Def* Mem2Reg::get_val(Lam* lam, const Analyze* proxy) {
    const auto& info = lam2info(lam);
    if (auto val = info.proxy2val.lookup(proxy)) {
        outf("get_val {} for {}: {}\n", lam, proxy, *val);
        return *val;
    }

    switch (info.lattice) {
        case Info::Preds0: return world().bot(proxy_type(proxy));
        case Info::Preds1: return get_val(info.pred, proxy);
        default: {
            auto old_lam = original(lam);
            outf("virtual phi: {}/{} for {}\n", old_lam, lam, proxy);
            return set_val(lam, proxy, world().analyze(proxy_type(proxy), {old_lam, proxy}, index(), {"phi"}));
        }
    }
}

const Def* Mem2Reg::set_val(Lam* lam, const Analyze* proxy, const Def* val) {
    outf("set_val {} for {}: {}\n", lam, proxy, val);
    return lam2info(lam).proxy2val[proxy] = val;
}

void Mem2Reg::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    // we need to install a phi in lam next time around
    if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == index() && analyze->op(1)->isa<Analyze>()) {
        auto phi_lam = analyze->op(0)->as_nominal<Lam>();
        auto proxy   = analyze->op(1)->as<Analyze>();
        auto proxy_lam = proxy->op(0)->as_nominal<Lam>();
        auto slot_id  = as_lit<u64>(proxy->op(1));

        auto& phi_info   = lam2info(phi_lam);
        auto& proxy_info = lam2info(proxy_lam);
        auto& phis = phi_info.phis;

        if (phi_info.lattice == Info::Keep) {
            if (proxy_info.slots[slot_id] == Info::SSA) {
                proxy_info.slots[slot_id] = Info::Keep_;
                outf("keep: {}\n", proxy);
                if (auto i = std::find(phis.begin(), phis.end(), proxy); i != phis.end())
                    phis.erase(i);
                man().undo(proxy_info.undo);
            }
        } else {
            assert(phi_info.lattice == Info::PredsN);
            assertf(std::find(phis.begin(), phis.end(), proxy) == phis.end(), "already added proxy {} to {}", proxy, phi_lam);
            //assert(proxy_info.undo <= lam_info.undo);
            phis.emplace_back(proxy);
            outf("phi needed: {}\n", analyze);
            man().undo(phi_info.undo);
        }
        return;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto proxy = isa_proxy(op)) {
            auto proxy_lam = proxy->op(0)->as_nominal<Lam>();
            auto slot_id  = as_lit<u64>(proxy->op(1));
            auto& info = lam2info(proxy_lam);
            if (info.slots[slot_id] == Info::SSA) {
                outf("keep: {}\n", proxy);
                info.slots[slot_id] = Info::Keep_;
                man().undo(info.undo);
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            if (lam->is_basicblock() && lam != man().cur_lam())
                lam2info(lam).writable.insert_range(range(lam2info(man().cur_lam()).writable));
            auto& info = lam2info(original(lam));
            auto pred = man().cur_lam();

            switch (info.lattice) {
                case Info::Preds0:
                    info.lattice = Info::Preds1;
                    info.pred = pred;
                    assert(info.phis.empty());
                    break;
                case Info::Preds1:
                    info.lattice = Info::PredsN;
                    outf("Preds1 -> PredsN\n");
                    man().undo(info.undo);
                    break;
                default:
                    break;
            }

            // if lam does not occur as callee and has more than one pred
            if ((!def->isa<App>() || i != 0) && (info.lattice == Info::PredsN )) {
                info.lattice = Info::Keep;
                outf("keep: {}\n", lam);
                for (auto phi : info.phis) {
                    auto& proxy_info = lam2info(phi->op(0)->as_nominal<Lam>());
                    auto slot_id    = as_lit<u64>(phi->op(1));
                    proxy_info.slots[slot_id] = Info::Keep_;
                    info.phis.clear();
                    man().undo(info.undo);
                    man().undo(proxy_info.undo);
                }
            }
        }
    }
}

}
