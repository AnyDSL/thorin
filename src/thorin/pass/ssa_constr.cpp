#include "thorin/pass/ssa_constr.h"

#include "thorin/util.h"

namespace thorin {

static const Def* get_sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static Lam* get_sloxy_lam(const Proxy* sloxy) { return sloxy->op(0)->as_nominal<Lam>(); }
static std::tuple<Lam*, const Proxy*> split_phixy(const Proxy* phixy) { return {phixy->op(0)->as_nominal<Lam>(), phixy->op(1)->as<Proxy>()}; }

const Proxy* SSAConstr::isa_sloxy(const Def* def) { if (auto p = isa_proxy(def); p && !p->op(1)->isa<Proxy>()) return p; return nullptr; }
const Proxy* SSAConstr::isa_phixy(const Def* def) { if (auto p = isa_proxy(def); p &&  p->op(1)->isa<Proxy>()) return p; return nullptr; }

// both sloxy and phixy reference the *old* lam
// the value map for get_val/set_val uses the *new* lam

void SSAConstr::inspect(Def*, Def* nom) {
    auto old_lam = nom->isa<Lam>();
    if (old_lam == nullptr) return;                           // not a lam
    if (old_lam->intrinsic() != Lam::Intrinsic::None) return; // its's an intrinsic
    if (old_lam->is_external()) return;                       // don't do sth with externals
    if (keep_.contains(old_lam)) return;                      // keep it as it is
    if (!preds_n_.contains(old_lam)) return;                  // doesn't have more than 1 preds

    // build a new_lam with phis as params
    if (auto& phis = lam2phis_[old_lam]; !phis.empty()) {
        std::vector<const Def*> types;
        for (auto i = phis.begin(), e = phis.end(); i != e;) {
            auto sloxy = *i;
            if (keep_.contains(sloxy)) {
                i = phis.erase(i);
            } else {
                types.emplace_back(get_sloxy_type(sloxy));
                ++i;
            }
        }

        auto& info = lam2info(old_lam);
        auto new_domain = merge_sigma(old_lam->domain(), types);
        auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
        world().DLOG("install phi for old_lam => new_lam: {}: {} => {}: {}",
                old_lam, old_lam->type()->domain(), new_lam, new_lam->type()->domain());
        info.new_lam = new_lam;
        new2old_[new_lam] = old_lam;
        preds_n_.emplace(new_lam);
    }
}

void SSAConstr::enter(Def* nom) {
    auto new_lam = nom->isa<Lam>();
    if (new_lam == nullptr) return;

    if (auto old_lam_opt = new2old_.lookup(new_lam)) {
        auto old_lam = *old_lam_opt;
        auto& phis = lam2phis_[old_lam];

        size_t n = new_lam->num_params() - phis.size();
        auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
        man().map(old_lam->param(), new_param);
        new_lam->set(old_lam->ops());

        size_t i = 0;
        for (auto phi : phis)
            set_val(new_lam, phi, new_lam->param(n + i++));
    }
}

const Def* SSAConstr::rewrite(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_lam == nullptr) return def;

    if (auto slot = isa<Tag::Slot>(def)) {
        auto [out_mem, out_ptr] = slot->split<2>();
        auto old_lam = new2old(cur_lam);
        auto slot_id = lam2info(old_lam).num_slots++;
        auto sloxy = proxy(out_ptr->type(), {old_lam, world().lit_nat(slot_id)}, slot->debug());
        if (!keep_.contains(sloxy)) {
            set_val(cur_lam, sloxy, world().bot(get_sloxy_type(sloxy)));
            //lam2info(cur_lam).writable.emplace(sloxy);
            return world().tuple({slot->arg(), sloxy});
        }
    } else if (auto load = isa<Tag::Load>(def)) {
        auto [mem, ptr] = load->args<2>();
        if (auto sloxy = isa_sloxy(ptr))
            return world().tuple({mem, get_val(cur_lam, sloxy)});
    } else if (auto store = isa<Tag::Store>(def)) {
        auto [mem, ptr, val] = store->args<3>();
        if (auto sloxy = isa_sloxy(ptr)) {
            //if (lam2info(cur_lam).writable.contains(sloxy)) {
                set_val(cur_lam, sloxy, val);
                return mem;
            //}
        }
    } else if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            if (auto new_lam = lam2info(lam).new_lam) {
                auto& phis = lam2phis_[lam];
                auto phi = phis.begin();
                Array<const Def*> args(phis.size(), [&](auto) { return get_val(cur_lam, *phi++); });
                return world().app(new_lam, merge_tuple(app->arg(), args));
            }
        }
    }

    return def;
}

const Def* SSAConstr::get_val(Lam* lam, const Proxy* sloxy) {
    const auto& info = lam2info(lam);

    if (auto val = info.sloxy2val.lookup(sloxy)) {
        world().DLOG("get_val {} for {}: {}", lam, sloxy, *val);
        return *val;
    } else if (preds_n_.contains(lam)) {
        auto old_lam = new2old(lam);
        world().DLOG("phixy: {}/{} for {}", old_lam, lam, sloxy);
        return set_val(lam, sloxy, proxy(get_sloxy_type(sloxy), {old_lam, sloxy}, {"phi"}));
    } else if (info.lattice == Info::Preds1) {
        world().DLOG("get_val pred: {}: {} -> {}", sloxy, lam, info.pred);
        return get_val(info.pred, sloxy);
    } else {
        assert(info.lattice == Info::Preds0);
        return world().bot(get_sloxy_type(sloxy));
    }
}

const Def* SSAConstr::set_val(Lam* lam, const Proxy* sloxy, const Def* val) {
    world().DLOG("set_val {} for {}: {}", lam, sloxy, val);
    return lam2info(lam).sloxy2val[sloxy] = val;
}

undo_t SSAConstr::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_lam == nullptr) return No_Undo;
    if (def->isa<Param>())  return No_Undo;
    if (isa_phixy(def))     return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto sloxy = isa_sloxy(op)) {
            auto sloxy_lam = get_sloxy_lam(sloxy);

            if (keep_.emplace(sloxy).second) {
                world().DLOG("keep: {}", sloxy);
                undo = std::min(undo, (undo_t) lam2info(sloxy_lam).undo);
            }
        } else if (auto phixy = isa_phixy(op)) { // we need to install a phi in lam next time around
            auto [phixy_lam, sloxy] = split_phixy(phixy);
            auto sloxy_lam = get_sloxy_lam(sloxy);
            auto& phis = lam2phis_[phixy_lam];

            if (keep_.contains(phixy_lam)) {
                if (keep_.emplace(sloxy).second) {
                    world().DLOG("keep: {}", sloxy);
                    if (auto i = phis.find(sloxy); i != phis.end()) phis.erase(i);
                    return lam2info(sloxy_lam).undo;
                }
            } else {
                phis.emplace(sloxy);
                world().DLOG("phi needed: {}", phixy);
                return lam2info(phixy_lam).undo;
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            //if (lam->is_basicblock() && lam != man().cur_lam())
                //lam2info(lam).writable.insert_range(range(lam2info(man().cur_lam()).writable));
            lam = new2old(lam);
            auto& info = lam2info(lam);
            auto& phis = lam2phis_[lam];

            if (preds_n_.contains(lam)) {
            } else if (info.lattice == Info::Preds1) {
                preds_n_.emplace(lam);
                world().DLOG("Preds1 -> PredsN: {}", lam);
                undo = std::min(undo, (undo_t) info.undo);
            } else if (info.lattice == Info::Preds0) {
                if (info.pred != cur_lam) {
                    info.lattice = Info::Preds1;
                    info.pred = cur_lam;
                    assert(phis.empty());
                }
            }

            // if lam does not occur as callee and has more than one pred
            if ((!def->isa<App>() || i != 0) && preds_n_.contains(lam)) {
                keep_.emplace(lam);
                world().DLOG("keep: {}", lam);
                for (auto phi : phis) {
                    auto sloxy_lam = get_sloxy_lam(phi);
                    auto& proxy_info = lam2info(sloxy_lam);
                    keep_.emplace(phi);
                    undo = std::min(undo, (undo_t)       info.undo);
                    undo = std::min(undo, (undo_t) proxy_info.undo);
                }
                phis.clear();
            }
        }
    }

    return undo;
}

}
