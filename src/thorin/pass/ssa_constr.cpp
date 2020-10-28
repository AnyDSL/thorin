#include "thorin/pass/ssa_constr.h"

#include "thorin/util.h"

namespace thorin {

static const Def* get_sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static Lam* get_sloxy_lam(const Proxy* sloxy) { return sloxy->op(0)->as_nominal<Lam>(); }
static std::tuple<const Proxy*, Lam*> split_phixy(const Proxy* phixy) { return {phixy->op(0)->as<Proxy>(), phixy->op(1)->as_nominal<Lam>()}; }

void SSAConstr::enter(Def* nom) {
    if (auto lam = nom->isa<Lam>()) {
        lam2sloxy2val_[lam].clear();
        slot_id_ = 0;
    }
}

const Def* SSAConstr::prewrite(Def* cur_nom, const Def* def) {
    if (auto cur_lam = cur_nom->isa<Lam>()) {
        if (auto traxy = isa_proxy(Traxy, def)) {
            world().DLOG("traxy '{}'", traxy);
            for (size_t i = 1, e = def->num_ops(); i != e; i += 2)
                set_val(cur_lam, as_proxy(Sloxy, traxy->op(i)), traxy->op(i+1));
            return traxy->op(0);
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
    } else if (auto app = def->isa<App>()) {
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

std::variant<const Def*, undo_t> SSAConstr::mem2phi(Lam* cur_lam, const App* app, Lam* mem_lam) {
    auto& phis = lam2phis_[mem_lam];
    if (phis.empty()) return app;

    get<Visit>(mem_lam); // create undo
    auto& phi_lam = mem2phi_.emplace(mem_lam, nullptr).first->second;

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

    size_t num_phis = phis.size();
    if (num_phis == 0) return app;

    if (phi_lam == nullptr) {
        auto new_type = world().pi(merge_sigma(mem_lam->domain(), types), mem_lam->codomain());
        phi_lam = world().lam(new_type, mem_lam->debug());
        world().DLOG("new phi_lam '{}'", phi_lam);

        man().mark_tainted(phi_lam);
        world().DLOG("mem_lam => phi_lam: '{}': '{}' => '{}': '{}'", mem_lam, mem_lam->type()->domain(), phi_lam, phi_lam->domain());
        preds_n_.emplace(phi_lam);

        auto num_mem_params = mem_lam->num_params();
        Array<const Def*> new_params(num_mem_params, [&](size_t i) { return phi_lam->param(i); });
        auto new_param = world().tuple(new_params);
        auto filter    = world().subst(mem_lam->filter(), mem_lam->param(), new_param);
        auto body      = world().subst(mem_lam->body(),   mem_lam->param(), new_param);

        size_t i = 0;
        Array<const Def*> traxy_ops(2*num_phis + 1);
        traxy_ops[0] = filter;
        for (auto phi : phis) {
            traxy_ops[2*i + 1] = phi;
            traxy_ops[2*i + 2] = phi_lam->param(num_mem_params + i);
            ++i;
        }

        phi_lam->set_filter(proxy(filter->type(), traxy_ops, Traxy));
        phi_lam->set_body(body);
    } else {
        world().DLOG("reuse phi_lam '{}'", phi_lam);
    }


    auto phi = phis.begin();
    Array<const Def*> args(phis.size(), [&](auto) { return get_val(cur_lam, *phi++); });
    return world().app(phi_lam, merge_tuple(app->arg(), args));
}

undo_t SSAConstr::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (!cur_lam || def->is_const() || analyzed(def) || def->isa<Param>() || def->isa_nominal()) return No_Undo;
    if (auto proxy = def->isa<Proxy>(); proxy && proxy->index() != index()) return No_Undo;

    if (auto sloxy = isa_proxy(Sloxy, def)) {
        auto sloxy_lam = get_sloxy_lam(sloxy);

        if (keep_.emplace(sloxy).second) {
            world().DLOG("keep: '{}'; pointer needed for: '{}'", sloxy, def);
            auto&& [_, undo, __] = get<Enter>(sloxy_lam);
            return undo;
        }
    } else if (auto phixy = isa_proxy(Phixy, def)) {
        auto [sloxy, mem_lam] = split_phixy(phixy);
        //auto sloxy_lam = get_sloxy_lam(sloxy);
        auto& phis = lam2phis_[mem_lam];

        if (phis.emplace(sloxy).second) {
            auto&& [_, undo, __] = get<Visit>(mem_lam);
            world().DLOG("phi needed: phixy '{}' for sloxy '{}' for mem_lam '{}' -> state {}", phixy, sloxy, mem_lam, undo);
            mem2phi_[mem_lam] = nullptr;
            return undo;
        }
    }

    auto undo = No_Undo;
    auto app = def->isa<App>();
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        undo = std::min(undo, analyze(cur_nom, def->op(i)));
        if (auto lam = def->op(i)->isa_nominal<Lam>()) {
            bool callee_pos = app != nullptr && i == 0;
            if (auto u = join(cur_lam, lam, callee_pos); u != No_Undo) undo = std::min(undo, u);
        }
    }

    return undo;
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
            } else {
                world().DLOG("preds_1_{}callee_pos join {}callee_pos -> keep: '{}' with pred '{}'",
                        visit.callee_pos ? "" : "non_", callee_pos ? "" : "non_", lam, cur_lam);
                keep_.emplace(lam);

                // TODO invalidate phis
            }
            return undo;
        }
    } else {
        if (!callee_pos) {
            world().DLOG("preds_n -> keep: {}", lam);
            keep_.emplace(lam);
            // TODO invalidate phis
            return undo;
        }
    }

    return No_Undo;
}

}
