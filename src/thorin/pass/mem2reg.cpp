#include "thorin/pass/mem2reg.h"

#include "thorin/util.h"
#include "thorin/util/log.h"

namespace thorin {

static const Def* proxy_type(const Analyze* proxy) { return proxy->type()->as<Ptr>()->pointee(); }
static std::tuple<Lam*, int64_t> disassemble_proxy(const Analyze* proxy) { return {proxy->op(1)->as_nominal<Lam>(), as_lit<u64>(proxy->op(2))}; }
static std::tuple<Lam*, const Analyze*> disassemble_virtual_phi(const Analyze* proxy) { return {proxy->op(1)->as_nominal<Lam>(), proxy->op(2)->as<Analyze>()}; }

const Analyze* Mem2Reg::isa_proxy(const Def* def) {
    if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == index() && !analyze->op(2)->isa<Analyze>()) return analyze;
    return nullptr;
}

const Analyze* Mem2Reg::isa_virtual_phi(const Def* def) {
    if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == index() && analyze->op(2)->isa<Analyze>()) return analyze;
    return nullptr;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        auto orig = original(man().cur_lam());
        auto& info = lam2info(orig);
        auto slot_id = info.num_slots++;
        auto proxy = world().analyze(slot->out_ptr()->type(), index(), {orig, world().lit_uint(slot_id)}, slot->debug());
        if (!keep_.contains(proxy)) {
            set_val(proxy, world().bot(proxy_type(proxy)));
            lam2info(man().cur_lam()).writable.emplace(proxy);
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
                auto& phis = lam2phis_[lam];
                auto phi = phis.begin();
                Array<const Def*> args(phis.size(), [&](auto) { return get_val(*phi++); });
                return world().app(new_lam, merge_tuple(app->arg(), args));
            }
        }
    }

    return def;
}

void Mem2Reg::inspect(Def* def) {
    if (auto old_lam = def->isa<Lam>()) {
        auto& info = lam2info(old_lam);
        if (preds_n_.contains(old_lam)) info.lattice = Info::PredsN;
        if (keep_   .contains(old_lam)) info.lattice = Info::Keep;

        if (old_lam->is_external() || old_lam->intrinsic() != Lam::Intrinsic::None) {
            info.lattice = Info::Keep;
        } else if (info.lattice != Info::Keep) {
            man().new_state();
            auto& info = lam2info(old_lam);
            auto& phis = lam2phis_[old_lam];

            if (info.lattice == Info::PredsN && !phis.empty()) {
                std::vector<const Def*> types;
                for (auto i = phis.begin(); i != phis.end();) {
                    auto proxy = *i;
                    if (keep_.contains(proxy)) {
                        i = phis.erase(i);
                    } else {
                        types.emplace_back(proxy_type(proxy));
                        ++i;
                    }
                }
                //Array<const Def*> types(phis.size(), [&](auto) { return proxy_type(*phi++); });
                auto new_domain = merge_sigma(old_lam->domain(), types);
                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                outf("new_lam: {} -> {}\n", old_lam, new_lam);
                new2old_[new_lam] = old_lam;
                info.new_lam = new_lam;
                lam2info(new_lam).lattice = Info::PredsN;
            }
        }
    }
}

void Mem2Reg::enter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        outf("enter: {}\n", new_lam);

        if (auto old_lam_opt = new2old_.lookup(new_lam)) {
            auto old_lam = *old_lam_opt;
            auto& phis = lam2phis_[old_lam];

            outf("enter: {}/{}\n", old_lam, new_lam);
            size_t n = new_lam->num_params() - phis.size();

            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
            man().map(old_lam->param(), new_param);
            new_lam->set(old_lam->ops());

            size_t i = 0;
            for (auto phi : phis)
                set_val(new_lam, phi, new_lam->param(n + i++));
        }
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
        case Info::Preds1:
                           outf("get_val pred: {}: {} -> {}\n", proxy, lam, info.pred);
                           return get_val(info.pred, proxy);
        default: {
            auto old_lam = original(lam);
            outf("virtual phi: {}/{} for {}\n", old_lam, lam, proxy);
            return set_val(lam, proxy, world().analyze(proxy_type(proxy), index(), {old_lam, proxy}, {"phi"}));
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
    if (auto phi = isa_virtual_phi(def)) {
        auto [phi_lam, proxy] = disassemble_virtual_phi(phi);
        auto [proxy_lam, slot_id] = disassemble_proxy(proxy);

        auto& phi_info   = lam2info(phi_lam);
        auto& proxy_info = lam2info(proxy_lam);
        auto& phis = lam2phis_[phi_lam];

        if (phi_info.lattice == Info::Keep) {
            if (keep_.emplace(proxy).second) {
                outf("keep: {}\n", proxy);
                if (auto i = phis.find(proxy); i != phis.end())
                    phis.erase(i);
                man().undo(proxy_info.undo);
            }
        } else {
            assert(phi_info.lattice == Info::PredsN);
            assertf(phis.find(proxy) == phis.end(), "already added proxy {} to {}", proxy, phi_lam);
            phis.emplace(proxy);
            outf("phi needed: {}\n", phi);
            man().undo(phi_info.undo);
        }
        return;
    } else if (isa_proxy(def)) {
        return;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto proxy = isa_proxy(op)) {
            auto [proxy_lam, slot_id] = disassemble_proxy(proxy);
            auto& info = lam2info(proxy_lam);
            if (keep_.emplace(proxy).second) {
                outf("keep: {}\n", proxy);
                man().undo(info.undo);
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            if (lam->is_basicblock() && lam != man().cur_lam())
                lam2info(lam).writable.insert_range(range(lam2info(man().cur_lam()).writable));
            auto orig = original(lam);
            auto& info = lam2info(orig);
            auto& phis = lam2phis_[orig];
            auto pred = man().cur_lam();

            switch (info.lattice) {
                case Info::Preds0:
                    info.lattice = Info::Preds1;
                    info.pred = pred;
                    assert(phis.empty());
                    break;
                case Info::Preds1:
                    info.lattice = Info::PredsN;
                    preds_n_.emplace(orig);
                    outf("Preds1 -> PredsN: {}\n", orig);
                    man().undo(info.undo);
                    break;
                default:
                    break;
            }

            // if lam does not occur as callee and has more than one pred
            if ((!def->isa<App>() || i != 0) && (info.lattice == Info::PredsN )) {
                info.lattice = Info::Keep;
                outf("keep: {}\n", lam);
                keep_.emplace(lam);
                for (auto phi : phis) {
                    auto [proxy_lam, slot_id] = disassemble_proxy(phi);
                    auto& proxy_info = lam2info(proxy_lam);
                    keep_.emplace(phi);
                    man().undo(info.undo);
                    man().undo(proxy_info.undo);
                }
                phis.clear();
            }
        }
    }
}

}
