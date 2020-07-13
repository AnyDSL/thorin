#include "thorin/pass/ssa_constr.h"

#include "thorin/util.h"
#include "thorin/analyses/scope.h"

namespace thorin {

/*
 * Proxies & helpers
 */

static const Def* get_sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static Lam* get_sloxy_lam(const Proxy* sloxy) { return sloxy->op(0)->as_nominal<Lam>(); }
static std::tuple<Lam*, const Proxy*> split_phixy(const Proxy* phixy) { return {phixy->op(0)->as_nominal<Lam>(), phixy->op(1)->as<Proxy>()}; }

const Proxy* SSAConstr::make_sloxy(Lam* lam, const Def* slot) {
    auto [out_mem, out_ptr] = slot->split<2>();
    auto sloxy = proxy(out_ptr->type(), {lam, world().lit_nat(slot_id_++)}, slot->debug());
    world().DLOG("sloxy: {}", sloxy);
    return sloxy;
}

const Proxy* SSAConstr::make_phixy(Lam* lam, const Proxy* sloxy) {
    auto phixy = proxy(get_sloxy_type(sloxy), {lam, sloxy}, sloxy->debug());
    phixy->set_name(std::string("phi_") + phixy->name());
    world().DLOG("phixy within {} for {}", lam, sloxy);
    return phixy;
}

const Proxy* SSAConstr::make_setxy(const Def* mem, const Proxy* sloxy, const Def* value) {
    auto setxy = proxy(get_sloxy_type(sloxy), {mem, sloxy, value});
    world().DLOG("setxy({}, {}, {})", mem, sloxy, value);
    return setxy;
}

const Proxy* SSAConstr::isa_sloxy(const Def* def) { if (auto p = isa_proxy(def); p && !p->op(1)->isa<Proxy>()) return p; return nullptr; }
const Proxy* SSAConstr::isa_phixy(const Def* def) { if (auto p = isa_proxy(def); p &&  p->op(1)->isa<Proxy>()) return p; return nullptr; }

/*
 * PassMan hooks
 */

const Def* SSAConstr::rewrite(Def* cur_nom, const Def* def) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto slot = isa<Tag::Slot>(def)) {
            auto sloxy = make_sloxy(cur_lam, slot);
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
            if (auto mem_lam = app->callee()->isa_nominal<Lam>()) return rewrite(cur_lam, app, mem_lam);
        }
    }

    return def;
}

const Def* SSAConstr::get_val(Lam* lam, const Proxy* sloxy) {
    if (auto val = sloxy2val_.lookup(sloxy)) {
        world().DLOG("get_val {} for {}: {}", lam, sloxy, *val);
        return *val;
    } else {
        auto&& [visit, _] = get<Visit>(lam);
        if (preds_n_.contains(lam)) {
            return set_val(lam, sloxy, make_phixy(lam, sloxy));
        } else if (visit.preds == Visit::Preds1) {
            world().DLOG("get_val pred: {}: {} -> {}", sloxy, lam, visit.pred);
            return get_val(visit.pred, sloxy);
        } else {
            assert(visit.preds == Visit::Preds0);
            return world().bot(get_sloxy_type(sloxy));
        }
    }
}

const Def* SSAConstr::set_val(Lam* lam, const Proxy* sloxy, const Def* val) {
    world().DLOG("set_val {} for {}: {}", lam, sloxy, val);
    return sloxy2val_[sloxy] = val;
}

const Def* SSAConstr::rewrite(Lam* /*cur_lam*/, const App* app, Lam* mem_lam) {
    if (mem_lam->is_external() || !mem_lam->is_set() || keep_.contains(mem_lam))
        return app;
#if 0
            auto&& [visit, _] = get<Visit>(mem_lam);
            if (auto& phi_lam = visit.phi_lam) {
                auto& phis = lam2phis_[mem_lam];
                auto phi = phis.begin();
                Array<const Def*> args(phis.size(), [&](auto) { return get_val(cur_lam, *phi++); });
                return world().app(phi_lam, merge_tuple(app->arg(), args));
#endif

    Scope scope(mem_lam);
    std::vector<const Proxy*> phis;

    for (auto [sloxy, _] : sloxy2val_) {
        if (scope.free().contains(sloxy)) phis.emplace_back(sloxy);
    }

    if (auto& phis = lam2phis_[mem_lam]; !phis.empty()) {
        // build a phi_lam with phis as params if we can're reuse an old one
        auto&& [visit, _] = get<Visit>(mem_lam);
        if (auto& visit_phi_lam = visit.phi_lam; !visit_phi_lam) {
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

            auto phi_domain = merge_sigma(mem_lam->domain(), types);
            auto new_type = world().pi(phi_domain, mem_lam->codomain());
            auto& phi_lam = man().reincarnate<Lam>(mem_lam);
            if (!phi_lam || phi_lam->type() != new_type) phi_lam = world().lam(new_type, mem_lam->debug());
            man().mark_tainted(phi_lam);
            world().DLOG("mem_lam => phi_lam: {}: {} => {}: {}", mem_lam, mem_lam->type()->domain(), phi_lam, phi_domain);
            preds_n_.emplace(phi_lam);
            phi2mem_[phi_lam] = mem_lam;
            visit_phi_lam = phi_lam;

            size_t n = phi_lam->num_params() - phis.size();
            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return phi_lam->param(i); }));
            phi_lam->set(0_s, world().subst(mem_lam->op(0), mem_lam->param(), new_param));
            phi_lam->set(1_s, world().subst(mem_lam->op(1), mem_lam->param(), new_param));

            size_t i = 0;
            for (auto phi : phis)
                set_val(phi_lam, phi, phi_lam->param(n + i++));
        }
    }

    return app;
}

undo_t SSAConstr::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (!cur_lam || def->isa<Param>() || isa_phixy(def)) return No_Undo;
    if (auto proxy = def->isa<Proxy>(); proxy && proxy->index() != index()) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto sloxy = isa_sloxy(op)) {
            // we can't SSA-construct this slot
            auto sloxy_lam = get_sloxy_lam(sloxy);

            if (keep_.emplace(sloxy).second) {
                world().DLOG("keep: {}; pointer needed for: {}", sloxy, def);
                auto&& [_, undo_enter] = get<Enter>(sloxy_lam);
                undo = std::min(undo, undo_enter);
            }
        } else if (auto phixy = isa_phixy(op)) {
            // we need to install a phi in lam next time around
            auto [phixy_lam, sloxy] = split_phixy(phixy);
            auto sloxy_lam = get_sloxy_lam(sloxy);
            auto& phis = lam2phis_[phixy_lam];

            if (keep_.contains(phixy)) {
                if (keep_.emplace(sloxy).second) {
                    world().DLOG("keep: {}; I can't adjust {}", sloxy, phixy_lam);
                    get<Visit>(phixy_lam).first.phi_lam = nullptr;
                    if (auto i = phis.find(sloxy); i != phis.end()) phis.erase(i);
                    auto&& [_, undo_visit] = get<Visit>(sloxy_lam);
                    return undo_visit;
                }
            } else {
                phis.emplace(sloxy);
                get<Visit>(phixy_lam).first.phi_lam = nullptr;
                world().DLOG("phi needed: phixy {} for sloxy {} for phixy_lam {}", phixy, sloxy, phixy_lam);
                auto&& [_, undo_visit] = get<Visit>(phixy_lam);
                return undo_visit;
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            //if (lam->is_basicblock() && lam != man().cur_lam())
                //lam2info(lam).writable.insert_range(range(lam2info(man().cur_lam()).writable));
            lam = lam2mem(lam);
            auto&& [visit, undo_visit] = get<Visit>(lam);
            auto& phis = lam2phis_[lam];

            if (preds_n_.contains(lam) || keep_.contains(lam) || lam->is_intrinsic() || lam->is_external()) {
            } else if (visit.preds == Visit::Preds1) {
                preds_n_.emplace(lam);
                world().DLOG("Preds1 -> PredsN: {}", lam);
                undo = std::min(undo, undo_visit);
            } else if (visit.preds == Visit::Preds0) {
                if (visit.pred != cur_lam) {
                    visit.pred = cur_lam;
                    visit.preds = Visit::Preds1;
                    assert(phis.empty());
                }
            }

            // if lam does not occur as callee and has more than one pred - we can't do anything
            if ((!def->isa<App>() || i != 0) && preds_n_.contains(lam)) {
                if (keep_.emplace(lam).second) {
                    world().DLOG("keep: {}", lam);

                    if (!phis.empty()) {
                        undo = std::min(undo, undo_visit);
                        phis.clear();
                    }
                }
            }
        }
    }

    return undo;
}

}
