#include "thorin/pass/ssa_constr.h"

#include "thorin/util.h"

namespace thorin {

/*
 * destruct proxies
 */

static const Def* get_sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static Lam* get_sloxy_lam(const Proxy* sloxy) { return sloxy->op(0)->as_nominal<Lam>(); }
static std::tuple<const Proxy*, Lam*> split_phixy(const Proxy* phixy) { return {phixy->op(0)->as<Proxy>(), phixy->op(1)->as_nominal<Lam>()}; }

/*
 * make proxies
 */

const Proxy* SSAConstr::make_sloxy(Lam* lam, const Def* slot) {
    auto [out_mem, out_ptr] = slot->split<2>();
    auto sloxy = proxy(out_ptr->type(), {lam, world().lit_nat(slot_id_++)}, Sloxy, slot->debug());
    world().DLOG("sloxy: {}", sloxy);
    return sloxy;
}

const Proxy* SSAConstr::make_traxy(const Def* mem, Lam* lam) {
    return proxy(mem->type(), {mem, lam}, Traxy, mem->debug());
}

/*
 * PassMan hooks
 */

void SSAConstr::enter(Def* nom) {
    slot_id_ = 0;
    pred_ = nullptr;
    if (auto lam = nom->isa<Lam>()) lam2sloxy2val_[lam].clear();
}

const Def* SSAConstr::prewrite(Def* cur_nom, const Def* def) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto traxy = isa_proxy(Traxy, def)) {
            pred_ = traxy->op(1)->as_nominal<Lam>();
            return traxy->op(0);
        } else if (auto stoxy = isa_proxy(Stoxy, def)) {
            auto [in, sloxy, val] = stoxy->ops<3>();
            set_val(cur_lam, sloxy->as<Proxy>(), val);
            return in;
        }
    }

    return def;
}

std::variant<const Def*, undo_t> SSAConstr::rewrite(Def* cur_nom, const Def* def) {
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
            if (auto sloxy = isa_proxy(Sloxy, ptr))
                return world().tuple({mem, get_val(cur_lam, sloxy)});
        } else if (auto store = isa<Tag::Store>(def)) {
            auto [mem, ptr, val] = store->args<3>();
            if (auto sloxy = isa_proxy(Sloxy, ptr)) {
                //if (lam2info(cur_lam).writable.contains(sloxy)) {
                    set_val(cur_lam, sloxy, val);
                    return mem;
                //}
            }
        }

        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            if (def->isa<Param>() || def->isa<Proxy>()) continue;
            if (auto lam = def->op(i)->isa_nominal<Lam>()) {
                if (!preds_n_.contains(lam)) {
                    auto&& [visit, undo, inserted] = get<Visit>(lam);
                    if (inserted) {
                        lam->set(1_s, make_traxy(lam->op(1), cur_lam)); // first case with: cur_lam -> lam
                    } else {
                        preds_n_.emplace(lam);
                        world().DLOG("Preds1 -> PredsN: {}", lam);
                        std::get<0>(get<Visit>(lam)).phi_lam = nullptr;
                        return undo;
                    }
                }
            }
        }

        if (auto app = def->isa<App>()) {
            if (auto mem_lam = app->callee()->isa_nominal<Lam>())
                return rewrite(cur_lam, app, mem_lam); // lam in callee postion potentially needs phis
        }
    }

    return def;
}

const Def* SSAConstr::get_val(Lam* lam, const Proxy* sloxy) {
    if (auto val = lam2sloxy2val_[lam].lookup(sloxy)) {
        world().DLOG("get_val found: {}: {}: {}", sloxy, *val, lam);
        return *val;
    } else if (pred_ != nullptr) {
        world().DLOG("get_val recurse: {}: {} -> {}", sloxy, pred_, lam);
        return get_val(pred_, sloxy);
    } else {
        auto phixy = proxy(get_sloxy_type(sloxy), {sloxy, lam}, Phixy, sloxy->debug());
        phixy->set_name(std::string("phi_") + phixy->name());
        world().DLOG("get_val phixy: {} {}", sloxy, lam);
        return set_val(lam, sloxy, phixy);
    }
}

const Def* SSAConstr::set_val(Lam* lam, const Proxy* sloxy, const Def* val) {
    world().DLOG("set_val: {}: {}: {}", sloxy, val, lam);
    return lam2sloxy2val_[lam][sloxy] = val;
}

std::variant<const Def*, undo_t> SSAConstr::rewrite(Lam* cur_lam, const App* app, Lam* mem_lam) {
    if (mem_lam->is_external() || !mem_lam->is_set() || keep_.contains(mem_lam))
        return app;

    if (auto& phis = lam2phis_[mem_lam]; !phis.empty()) {
        // build a phi_lam with phis as params if we can're reuse an old one
        auto&& [visit, _, __] = get<Visit>(mem_lam);
        auto& phi_lam = visit.phi_lam;
        if (phi_lam == nullptr) {
            std::vector<const Def*> types;
            std::vector<const Def*> args;
            for (auto i = phis.begin(), e = phis.end(); i != e;) {
                auto sloxy = *i;
                if (keep_.contains(sloxy)) {
                    i = phis.erase(i);
                } else {
                    types.emplace_back(get_sloxy_type(sloxy));
                    ++i;
                }
            }

            if (!phi_lam) {
                auto phi_domain = merge_sigma(mem_lam->domain(), types);
                auto new_type = world().pi(phi_domain, mem_lam->codomain());
                phi_lam = world().lam(new_type, mem_lam->debug());
            }

            man().mark_tainted(phi_lam);
            world().DLOG("mem_lam => phi_lam: {}: {} => {}: {}", mem_lam, mem_lam->type()->domain(), phi_lam, phi_lam->domain());
            preds_n_.emplace(phi_lam);
            phi2mem_[phi_lam] = mem_lam;

            size_t n = phi_lam->num_params() - phis.size();
            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return phi_lam->param(i); }));
            phi_lam->set(0_s, world().subst(mem_lam->op(0), mem_lam->param(), new_param));
            phi_lam->set(1_s, world().subst(mem_lam->op(1), mem_lam->param(), new_param));

            size_t i = 0;
            for (auto phi : phis) {
                auto stoxy = proxy(phi_lam->op(1)->type(), {phi_lam->op(1), phi, phi_lam->param(n + i++)}, Stoxy, phi->debug());
                phi_lam->set(1_s, stoxy);
            }
        }

        auto phi = phis.begin();
        Array<const Def*> args(phis.size(), [&](auto) { return get_val(cur_lam, *phi++); });
        return world().app(phi_lam, merge_tuple(app->arg(), args));
    }

    return app;
}

undo_t SSAConstr::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (!cur_lam || def->isa<Param>() || isa_proxy(Phixy, def)) return No_Undo;
    if (auto proxy = def->isa<Proxy>(); proxy && proxy->index() != index()) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto sloxy = isa_proxy(Sloxy, op)) {
            // we can't SSA-construct this slot
            auto sloxy_lam = get_sloxy_lam(sloxy);

            if (keep_.emplace(sloxy).second) {
                world().DLOG("keep: {}; pointer needed for: {}", sloxy, def);
                auto&& [_, undo_enter, __] = get<Enter>(sloxy_lam);
                undo = std::min(undo, undo_enter);
            }
        } else if (auto phixy = isa_proxy(Phixy, op)) {
            // we need to install a phi in lam next time around
            auto [sloxy, phixy_lam] = split_phixy(phixy);
            auto sloxy_lam = get_sloxy_lam(sloxy);
            auto& phis = lam2phis_[phixy_lam];

            if (keep_.contains(phixy)) {
                if (keep_.emplace(sloxy).second) {
                    world().DLOG("keep: {}; I can't adjust {}", sloxy, phixy_lam);
                    //get<Visit>(phixy_lam).first.phi_lam = nullptr; TODO
                    if (auto i = phis.find(sloxy); i != phis.end()) phis.erase(i);
                    auto&& [_, __, undo_visit] = get<Visit>(sloxy_lam);
                    return undo_visit;
                }
            } else {
                phis.emplace(sloxy);
                std::get<0>(get<Visit>(phixy_lam)).phi_lam = nullptr;
                world().DLOG("phi needed: phixy {} for sloxy {} for phixy_lam {}", phixy, sloxy, phixy_lam);
                auto&& [_, undo_visit, __] = get<Visit>(phixy_lam);
                return undo_visit;
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            //if (lam->is_basicblock() && lam != man().cur_lam())
                //lam2info(lam).writable.insert_range(range(lam2info(man().cur_lam()).writable));
            lam = lam2mem(lam);
            auto&& [visit, undo_visit, __] = get<Visit>(lam);
            auto& phis = lam2phis_[lam];

#if 0
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
#endif

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
