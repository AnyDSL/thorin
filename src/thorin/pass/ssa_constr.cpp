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
 * PassMan hooks
 */

void SSAConstr::enter(Def* nom) {
    if (auto lam = nom->isa<Lam>()) {
        lam2sloxy2val_[lam].clear();
        slot_id_ = 0;
    }
}

const Def* SSAConstr::prewrite(Def* cur_nom, const Def* def) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto stoxy = isa_proxy(Stoxy, def)) {
            for (size_t i = 1, e = def->num_ops(); i != e; i += 2)
                set_val(cur_lam, as_proxy(Sloxy, stoxy->op(i)), stoxy->op(i+1));
            return stoxy->op(0);
        }
    }

    return def;
}

std::variant<const Def*, undo_t> SSAConstr::rewrite(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_lam == nullptr || def->isa<Param>() || def->isa<Proxy>()) return def;

    if (auto slot = isa<Tag::Slot>(def)) {
        auto [out_mem, out_ptr] = slot->split<2>();
        auto sloxy = proxy(out_ptr->type(), {cur_lam, world().lit_nat(slot_id_++)}, Sloxy, slot->debug());
        world().DLOG("sloxy: '{}'", sloxy);
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

    auto app = def->isa<App>();
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto lam = def->op(i)->isa_nominal<Lam>()) {
            bool callee_pos = app != nullptr && i == 0;
            if (auto undo = join(cur_lam, lam, callee_pos); undo != No_Undo) return undo;
        }
    }

    if (app) {
        if (auto mem_lam = app->callee()->isa_nominal<Lam>(); mem_lam != nullptr && !keep(mem_lam))
            return mem2phi(cur_lam, app, mem_lam);
    }

    return def;
}

const Def* SSAConstr::get_val(Lam* lam, const Proxy* sloxy) {
    if (auto val = lam2sloxy2val_[lam].lookup(sloxy)) {
        world().DLOG("get_val found: '{}': '{}': '{}'", sloxy, *val, lam);
        return *val;
    } else if (auto pred = std::get<0>(get<Visit>(lam)).pred) {
        world().DLOG("get_val recurse: '{}': '{}' -> '{}'", sloxy, pred, lam);
        return get_val(pred, sloxy);
    } else {
        auto phixy = proxy(get_sloxy_type(sloxy), {sloxy, lam}, Phixy, sloxy->debug());
        phixy->set_name(std::string("phi_") + phixy->name());
        world().DLOG("get_val phixy: '{}' '{}'", sloxy, lam);
        return set_val(lam, sloxy, phixy);
    }
}

const Def* SSAConstr::set_val(Lam* lam, const Proxy* sloxy, const Def* val) {
    world().DLOG("set_val: '{}': '{}': '{}'", sloxy, val, lam);
    return lam2sloxy2val_[lam][sloxy] = val;
}

undo_t SSAConstr::join(Lam* cur_lam, Lam* lam, bool callee_pos) {
    if (keep(lam)) return No_Undo;

    auto&& [visit, undo, inserted] = get<Visit>(lam);
    if (!preds_n_.contains(lam)) {
        if (inserted) {
            visit.callee_pos = callee_pos;
            visit.pred = cur_lam;
            world().DLOG("preds_1_{}callee_pos '{}' with pred '{}'", callee_pos ? "" : "non_", lam, visit.pred);
        } else {
            if (visit.callee_pos && callee_pos) {
                preds_n_.emplace(lam);
                world().DLOG("preds_1_callee_pos -> preds_n: '{}'", lam);
                visit.phi_lam = nullptr;
            } else {
                world().DLOG("preds_1_{}callee_pos join {}callee_pos -> keep: '{}'",
                        visit.callee_pos ? "" : "non_", callee_pos ? "" : "non_", lam);
                keep_.emplace(lam);
            }
            return undo;
        }
    } else {
        if (!callee_pos) {
            keep_.emplace(lam);
            world().DLOG("preds_n -> keep: {}", lam);
            return undo;
        }
    }

    return No_Undo;
}

std::variant<const Def*, undo_t> SSAConstr::mem2phi(Lam* cur_lam, const App* app, Lam* mem_lam) {
    auto& phis = lam2phis_[mem_lam];
    if (phis.empty()) return app;

    auto&& [visit, _, __] = get<Visit>(mem_lam);
    auto& phi_lam = visit.phi_lam;
    if (phi_lam == nullptr) {
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

        if (phi_lam == nullptr) {
            auto new_type = world().pi(merge_sigma(mem_lam->domain(), types), mem_lam->codomain());
            phi_lam = world().lam(new_type, mem_lam->debug());
        }

        man().mark_tainted(phi_lam);
        world().DLOG("mem_lam => phi_lam: '{}': '{}' => '{}': '{}'", mem_lam, mem_lam->type()->domain(), phi_lam, phi_lam->domain());
        preds_n_.emplace(phi_lam);

        auto num_mem_params = mem_lam->num_params();
        Array<const Def*> new_params(num_mem_params, [&](size_t i) { return phi_lam->param(i); });
        auto new_param = world().tuple(new_params);
        auto filter    = world().subst(mem_lam->op(0), mem_lam->param(), new_param);
        auto body      = world().subst(mem_lam->op(1), mem_lam->param(), new_param);

        size_t i = 0, num_phis = phis.size();
        Array<const Def*> stoxy_ops(2*num_phis + 1);
        stoxy_ops[0] = filter;
        for (auto phi : phis) {
            stoxy_ops[2*i + 1] = phi;
            stoxy_ops[2*i + 2] = phi_lam->param(num_mem_params + i);
            ++i;
        }

        phi_lam->set_filter(proxy(filter->type(), stoxy_ops, Stoxy));
        phi_lam->set_body(body);
    }

    auto phi = phis.begin();
    Array<const Def*> args(phis.size(), [&](auto) { return get_val(cur_lam, *phi++); });
    return world().app(phi_lam, merge_tuple(app->arg(), args));
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
                world().DLOG("keep: '{}'; pointer needed for: '{}'", sloxy, def);
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
                    world().DLOG("keep: '{}'; I can't adjust '{}'", sloxy, phixy_lam);
                    //get<Visit>(phixy_lam).first.phi_lam = nullptr; TODO
                    if (auto i = phis.find(sloxy); i != phis.end()) phis.erase(i);
                    auto&& [_, __, undo_visit] = get<Visit>(sloxy_lam);
                    return undo_visit;
                }
            } else {
                phis.emplace(sloxy);
                std::get<0>(get<Visit>(phixy_lam)).phi_lam = nullptr;
                world().DLOG("phi needed: phixy '{}' for sloxy '{}' for phixy_lam '{}'", phixy, sloxy, phixy_lam);
                auto&& [_, undo_visit, __] = get<Visit>(phixy_lam);
                return undo_visit;
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            //if (lam->is_basicblock() && lam != man().cur_lam())
                //lam2info(lam).writable.insert_range(range(lam2info(man().cur_lam()).writable));
            //lam = lam2mem(lam);
            auto&& [visit, undo_visit, __] = get<Visit>(lam);
            auto& phis = lam2phis_[lam];

#if 0
            if (preds_n_.contains(lam) || keep_.contains(lam) || lam->is_intrinsic() || lam->is_external()) {
            } else if (visit.preds == Visit::Preds1) {
                preds_n_.emplace(lam);
                world().DLOG("Preds1 -> PredsN: '{}'", lam);
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
                    world().DLOG("keep: '{}'", lam);

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
