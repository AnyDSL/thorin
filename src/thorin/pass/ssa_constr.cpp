#include "thorin/pass/ssa_constr.h"

#include "thorin/util.h"

namespace thorin {

/*
 * helpers
 */

static const Def* sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static std::tuple<Lam*, int64_t     > split_sloxy(const Proxy* sloxy) { return {sloxy->op(0)->as_nominal<Lam>(), as_lit<u64>(sloxy->op(1))}; }
static std::tuple<Lam*, const Proxy*> split_phixy(const Proxy* proxy) { return {proxy->op(0)->as_nominal<Lam>(), proxy->op(1)->as<Proxy>()}; }

const Proxy* SSAConstr::isa_sloxy(const Def* def) { if (auto p = isa_proxy(def); p && !p->op(1)->isa<Proxy>()) return p; return nullptr; }
const Proxy* SSAConstr::isa_phixy(const Def* def) { if (auto p = isa_proxy(def); p &&  p->op(1)->isa<Proxy>()) return p; return nullptr; }

/*
 * PassMan hooks
 */

void SSAConstr::inspect(Def*, Def* nom) {
    if (auto old_lam = nom->isa<Lam>()) {
        auto& info = lam2info(old_lam);
        if (preds_n_.contains(old_lam)) info.lattice = Info::PredsN;
        if (keep_   .contains(old_lam)) info.lattice = Info::Keep;

        if (old_lam->is_external() || old_lam->intrinsic() != Lam::Intrinsic::None) {
            info.lattice = Info::Keep;
        } else if (info.lattice != Info::Keep) {
            auto& info = lam2info(old_lam);
            auto& phis = lam2phis_[old_lam];

            if (info.lattice == Info::PredsN && !phis.empty()) {
                std::vector<const Def*> types;
                for (auto i = phis.begin(); i != phis.end();) {
                    auto proxy = *i;
                    if (keep_.contains(proxy)) {
                        i = phis.erase(i);
                    } else {
                        types.emplace_back(sloxy_type(proxy));
                        ++i;
                    }
                }
                //Array<const Def*> types(phis.size(), [&](auto) { return sloxy_type(*phi++); });
                auto new_domain = merge_sigma(old_lam->domain(), types);
                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                world().DLOG("new_lam: {} -> {}", old_lam, new_lam);
                new2old_[new_lam] = old_lam;
                info.new_lam = new_lam;
                lam2info(new_lam).lattice = Info::PredsN;
            }
        }
    }
}

void SSAConstr::enter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        world().DLOG("enter: {}", new_lam);

        if (auto old_lam_opt = new2old_.lookup(new_lam)) {
            auto old_lam = *old_lam_opt;
            if (!old_lam->is_set()) return;

            auto& phis = lam2phis_[old_lam];

            world().DLOG("enter: {}/{}", old_lam, new_lam);
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

const Def* SSAConstr::rewrite(Def* cur_nom, const Def* def) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto slot = isa<Tag::Slot>(def)) {
            auto [out_mem, out_ptr] = slot->split<2>();
            auto orig = original(cur_lam);
            auto& info = lam2info(orig);
            auto slot_id = info.num_slots++;
            auto sloxy = proxy(out_ptr->type(), {orig, world().lit_nat(slot_id)}, slot->debug());
            if (!keep_.contains(sloxy)) {
                set_val(cur_lam, sloxy, world().bot(sloxy_type(sloxy)));
                lam2info(cur_lam).writable.emplace(sloxy);
                return world().tuple({slot->arg(), sloxy});
            }
        } else if (auto load = isa<Tag::Load>(def)) {
            auto [mem, ptr] = load->args<2>();
            if (auto sloxy = isa_sloxy(ptr))
                return world().tuple({mem, get_val(cur_lam, sloxy)});
        } else if (auto store = isa<Tag::Store>(def)) {
            auto [mem, ptr, val] = store->args<3>();
            if (auto sloxy = isa_proxy(ptr)) {
                if (lam2info(cur_lam).writable.contains(sloxy)) {
                    set_val(cur_lam, sloxy, val);
                    return mem;
                }
            }
        } else if (auto app = def->isa<App>()) {
            if (auto lam = app->callee()->isa_nominal<Lam>()) {
                const auto& info = lam2info(lam);
                if (auto new_lam = info.new_lam) {
                    auto& phis = lam2phis_[lam];
                    auto phi = phis.begin();
                    Array<const Def*> args(phis.size(), [&](auto) { return get_val(cur_lam, *phi++); });
                    return world().app(new_lam, merge_tuple(app->arg(), args));
                }
            }
        }
    }

    return def;
}

const Def* SSAConstr::get_val(Lam* lam, const Proxy* sloxy) {
    const auto& info = lam2info(lam);
    if (auto val = info.proxy2val.lookup(sloxy)) {
        world().DLOG("get_val {} for {}: {}", lam, sloxy, *val);
        return *val;
    }

    switch (info.lattice) {
        case Info::Preds0: return world().bot(sloxy_type(sloxy));
        case Info::Preds1:
                           world().DLOG("get_val pred: {}: {} -> {}", sloxy, lam, info.pred);
                           return get_val(info.pred, sloxy);
        default: {
            auto old_lam = original(lam);
            world().DLOG("phixy: {}/{} for {}", old_lam, lam, sloxy);
            return set_val(lam, sloxy, proxy(sloxy_type(sloxy), {old_lam, sloxy}, {"phi"}));
        }
    }
}

const Def* SSAConstr::set_val(Lam* lam, const Proxy* sloxy, const Def* val) {
    world().DLOG("set_val {} for {}: {}", lam, sloxy, val);
    return lam2info(lam).proxy2val[sloxy] = val;
}

size_t SSAConstr::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_lam == nullptr) return No_Undo;
    if (def->isa<Param>()) return No_Undo;

    // we need to install a phi in lam next time around
    if (auto phi = isa_phixy(def)) {
        auto [phi_lam, sloxy] = split_phixy(phi);
        auto [sloxy_lam, slot_id] = split_sloxy(sloxy);

        auto& phi_info   = lam2info(phi_lam);
        auto& sloxy_info = lam2info(sloxy_lam);
        auto& phis = lam2phis_[phi_lam];

        if (phi_info.lattice == Info::Keep) {
            if (keep_.emplace(sloxy).second) {
                world().DLOG("keep: {}", sloxy);
                if (auto i = phis.find(sloxy); i != phis.end())
                    phis.erase(i);
                return sloxy_info.undo;
            }
        } else {
            assert(phi_info.lattice == Info::PredsN);
            assertf(phis.find(sloxy) == phis.end(), "already added sloxy {} to {}", sloxy, phi_lam);
            phis.emplace(sloxy);
            world().DLOG("phi needed: {}", phi);
            return phi_info.undo;
        }
    } else if (isa_sloxy(def)) {
        return No_Undo;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto sloxy = isa_sloxy(op)) {
            auto [sloxy_lam, slot_id] = split_sloxy(sloxy);
            auto& info = lam2info(sloxy_lam);
            if (keep_.emplace(sloxy).second) {
                world().DLOG("keep: {}", sloxy);
                return info.undo;
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            if (lam->is_basicblock() && lam != cur_lam)
                lam2info(lam).writable.insert_range(range(lam2info(cur_lam).writable));
            auto orig = original(lam);
            auto& info = lam2info(orig);
            auto& phis = lam2phis_[orig];
            auto pred = cur_lam;

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
                    return info.undo;
                default:
                    break;
            }

            // if lam does not occur as callee and has more than one pred
            if ((!def->isa<App>() || i != 0) && (info.lattice == Info::PredsN )) {
                info.lattice = Info::Keep;
                world().DLOG("keep: {}", lam);
                keep_.emplace(lam);
                for (auto phi : phis) {
                    auto [sloxy_lam, slot_id] = split_sloxy(phi);
                    auto& sloxy_info = lam2info(sloxy_lam);
                    keep_.emplace(phi);
                    return std::min(info.undo, sloxy_info.undo);
                }
                phis.clear();
            }
        }
    }

    return No_Undo;
}

}
