#include "thorin/pass/mem2reg.h"

#include "thorin/util.h"

namespace thorin {

static const Def* proxy_type(const Analyze* proxy) { return as<Tag::Ptr>(proxy->type())->arg(0); }
static std::tuple<Lam*, const Analyze*> split_virtual_phi(const Analyze* proxy) { return {proxy->op(0)->as_nominal<Lam>(), proxy->op(1)->as<Analyze>()}; }

const Analyze* Mem2Reg::isa_proxy(const Def* def) {
    if (auto analyze = isa<Analyze>(index(), def); analyze && !analyze->op(1)->isa<Analyze>()) return analyze;
    return nullptr;
}

const Analyze* Mem2Reg::isa_virtual_phi(const Def* def) {
    if (auto analyze = isa<Analyze>(index(), def); analyze && analyze->op(1)->isa<Analyze>()) return analyze;
    return nullptr;
}

bool Mem2Reg::enter_scope(Def* def) { return def->isa<Lam>();}

bool Mem2Reg::enter_nominal(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        world().DLOG("enter: {}", new_lam);

        if (auto old_lam_opt = new2old_.lookup(new_lam)) {
            auto old_lam = *old_lam_opt;

            auto& phis = lam2phis_[old_lam];

            world().DLOG("enter: {}/{}", old_lam, new_lam);
            size_t n = new_lam->num_params() - phis.size();

            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
            man().scope_map(old_lam->param(), new_param);
            new_lam->set(old_lam->ops());

            size_t i = 0;
            for (auto phi : phis)
                set_val(new_lam, phi, new_lam->param(n + i++));
        }

        return true;
    }

    return false;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = isa<Tag::Slot>(def)) {
        auto [out_mem, out_ptr] = slot->split<2>();
        auto orig = original(man().new_nom<Lam>());
        auto slot_id = lam2info_[orig].num_slots++;
        auto proxy = world().analyze(out_ptr->type(), {orig, world().lit_nat(slot_id)}, index(), slot->debug());
        if (!keep_.contains(proxy)) {
            set_val(proxy, world().bot(proxy_type(proxy)));
            lam2info_[man().new_nom<Lam>()].writable.emplace(proxy);
            return world().tuple({slot->arg(), proxy});
        }
    } else if (auto load = isa<Tag::Load>(def)) {
        auto [mem, ptr] = load->args<2>();
        if (auto proxy = isa_proxy(ptr))
            return world().tuple({mem, get_val(proxy)});
    } else if (auto store = isa<Tag::Store>(def)) {
        auto [mem, ptr, val] = store->args<3>();
        if (auto proxy = isa_proxy(ptr)) {
            if (lam2info_[man().new_nom<Lam>()].writable.contains(proxy)) {
                set_val(proxy, val);
                return mem;
            }
        }
    } else if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            const auto& info = lam2info_[lam];
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
        auto& info = lam2info_[old_lam];
        if (preds_n_.contains(old_lam)) info.lattice = Info::PredsN;
        if (keep_   .contains(old_lam)) info.lattice = Info::Keep;

        if (old_lam->is_external() || old_lam->intrinsic() != Lam::Intrinsic::None) {
            info.lattice = Info::Keep;
        } else if (info.lattice != Info::Keep) {
            auto& info = lam2info_[old_lam];
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
                world().DLOG("new_lam: {} -> {}", old_lam, new_lam);
                new2old_[new_lam] = old_lam;
                info.new_lam = new_lam;
                lam2info_[new_lam].lattice = Info::PredsN;
            }
        }
    }
}

const Def* Mem2Reg::get_val(Lam* lam, const Analyze* proxy) {
    const auto& info = lam2info_[lam];
    if (auto val = info.proxy2val.lookup(proxy)) {
        world().DLOG("get_val {} for {}: {}", lam, proxy, *val);
        return *val;
    }

    switch (info.lattice) {
        case Info::Preds0: return world().bot(proxy_type(proxy));
        case Info::Preds1:
                           world().DLOG("get_val pred: {}: {} -> {}", proxy, lam, info.pred);
                           return get_val(info.pred, proxy);
        default: {
            auto old_lam = original(lam);
            world().DLOG("virtual phi: {}/{} for {}", old_lam, lam, proxy);
            return set_val(lam, proxy, world().analyze(proxy_type(proxy), {old_lam, proxy}, index(), {"phi"}));
        }
    }
}

const Def* Mem2Reg::set_val(Lam* lam, const Analyze* proxy, const Def* val) {
    world().DLOG("set_val {} for {}: {}", lam, proxy, val);
    return lam2info_[lam].proxy2val[proxy] = val;
}

bool Mem2Reg::analyze(const Def* def) {
    world().DLOG("analyze: {}", def);
    if (def->isa<Param>()) return true;

    // we need to install a phi in lam next time around
    if (auto phi = isa_virtual_phi(def)) {
        auto [phi_lam, proxy] = split_virtual_phi(phi);

        auto& phi_info   = lam2info_[phi_lam];
        auto& phis = lam2phis_[phi_lam];

        if (phi_info.lattice == Info::Keep) {
            if (keep_.emplace(proxy).second) {
                world().DLOG("keep: {}", proxy);
                if (auto i = phis.find(proxy); i != phis.end())
                    phis.erase(i);
            }
        } else {
            assert(phi_info.lattice == Info::PredsN);
            assertf(phis.find(proxy) == phis.end(), "already added proxy {} to {}", proxy, phi_lam);
            phis.emplace(proxy);
            world().DLOG("phi needed: {}", phi);
        }
        return false;
    } else if (isa_proxy(def)) {
        return true;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto proxy = isa_proxy(op)) {
            if (keep_.emplace(proxy).second) {
                world().DLOG("keep: {}", proxy);
                return false;
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            if (lam->is_basicblock() && lam != man().new_nom<Lam>())
                lam2info_[lam].writable.insert_range(range(lam2info_[man().new_nom<Lam>()].writable));
            auto orig = original(lam);
            auto& info = lam2info_[orig];
            auto& phis = lam2phis_[orig];
            auto pred = man().new_nom<Lam>();

            switch (info.lattice) {
                case Info::Preds0:
                    info.lattice = Info::Preds1;
                    info.pred = pred;
                    assert(phis.empty());
                    break;
                case Info::Preds1:
                    info.lattice = Info::PredsN;
                    preds_n_.emplace(orig);
                    world().DLOG("Preds1 -> PredsN: {}", orig);
                    return false;
                default:
                    break;
            }

            // if lam does not occur as callee and has more than one pred
            if ((!def->isa<App>() || i != 0) && (info.lattice == Info::PredsN )) {
                info.lattice = Info::Keep;
                world().DLOG("keep: {}", lam);
                keep_.emplace(lam);
                for (auto phi : phis) keep_.emplace(phi);
                phis.clear();
                return false;
            }
        }
    }

    return true;
}

void Mem2Reg::retry() {
    lam2info_.clear();
    new2old_.clear();
    lam2phis_.clear();
}

void Mem2Reg::clear() {
    retry();
    keep_.clear();
    preds_n_.clear();
}

}
