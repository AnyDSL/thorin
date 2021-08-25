#include "thorin/pass/fp/ssa_constr.h"

namespace thorin {

static const Def* get_sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static Lam* get_sloxy_lam(const Proxy* sloxy) { return sloxy->op(0)->as_nom<Lam>(); }
static std::tuple<const Proxy*, Lam*> split_phixy(const Proxy* phixy) { return {phixy->op(0)->as<Proxy>(), phixy->op(1)->as_nom<Lam>()}; }

void SSAConstr::enter() {
    data(cur_nom()).enter_undo = cur_undo();
    lam2sloxy2val_[cur_nom()].clear();
}

const Def* SSAConstr::rewrite(const Proxy* proxy) {
    if (auto traxy = isa_proxy(proxy, Traxy)) {
        world().DLOG("traxy '{}'", traxy);
        for (size_t i = 1, e = traxy->num_ops(); i != e; i += 2)
            set_val(cur_nom(), as_proxy(traxy->op(i), Sloxy), traxy->op(i+1));
        return traxy->op(0);
    }

    return proxy;
}

const Def* SSAConstr::rewrite(const Def* def) {
    if (auto slot = isa<Tag::Slot>(def)) {
        auto [mem, id] = slot->args<2>();
        auto [_, ptr] = slot->split<2>();
        auto sloxy = proxy(ptr->type(), {cur_nom(), id}, Sloxy, slot->dbg());
        world().DLOG("sloxy: '{}'", sloxy);
        if (!keep_.contains(sloxy)) {
            set_val(cur_nom(), sloxy, world().bot(get_sloxy_type(sloxy)));
            data(cur_nom()).writable.emplace(sloxy);
            return world().tuple({mem, sloxy});
        }
    } else if (auto load = isa<Tag::Load>(def)) {
        auto [mem, ptr] = load->args<2>();
        if (auto sloxy = isa_proxy(ptr, Sloxy))
            return world().tuple({mem, get_val(cur_nom(), sloxy)});
    } else if (auto store = isa<Tag::Store>(def)) {
        auto [mem, ptr, val] = store->args<3>();
        if (auto sloxy = isa_proxy(ptr, Sloxy)) {
            if (data(cur_nom()).writable.contains(sloxy)) {
                set_val(cur_nom(), sloxy, val);
                return mem;
            }
        }
    } else if (auto app = def->isa<App>()) {
        if (auto mem_lam = app->callee()->isa_nom<Lam>(); !ignore(mem_lam))
            return mem2phi(app, mem_lam);
    }

    return def;
}

const Def* SSAConstr::get_val(Lam* lam, const Proxy* sloxy) {
    if (auto val = lam2sloxy2val_[lam].lookup(sloxy)) {
        world().DLOG("get_val found: '{}': '{}': '{}'", sloxy, *val, lam);
        return *val;
    } else if (ignore(lam)) {
        world().DLOG("cannot install phi for '{}' in '{}'", sloxy, lam);
        return sloxy;
    } else if (auto pred = data(lam).pred) {
        world().DLOG("get_val recurse: '{}': '{}' -> '{}'", sloxy, pred, lam);
        return get_val(pred, sloxy);
    } else {
        auto phixy = proxy(get_sloxy_type(sloxy), {sloxy, lam}, Phixy, sloxy->dbg());
        phixy->set_name(std::string("phi_") + phixy->debug().name);
        world().DLOG("get_val phixy: '{}' '{}'", sloxy, lam);
        return set_val(lam, sloxy, phixy);
    }
}

const Def* SSAConstr::set_val(Lam* lam, const Proxy* sloxy, const Def* val) {
    world().DLOG("set_val: '{}': '{}': '{}'", sloxy, val, lam);
    return lam2sloxy2val_[lam][sloxy] = val;
}

const Def* SSAConstr::mem2phi(const App* app, Lam* mem_lam) {
    auto&& mem_info = data(mem_lam);
    if (mem_info.visit_undo == No_Undo) mem_info.visit_undo = cur_undo();

    auto&& lam2phixys = lam2phixys_[mem_lam];
    if (lam2phixys.empty()) return app;

    auto&& [_, phi_lam] = *mem2phi_.emplace(mem_lam, nullptr).first;
    std::vector<const Def*> types;
    for (auto i = lam2phixys.begin(), e = lam2phixys.end(); i != e;) {
        auto sloxy = *i;
        if (keep_.contains(sloxy)) {
            i = lam2phixys.erase(i);
            phi_lam = nullptr;
        } else {
            types.emplace_back(get_sloxy_type(sloxy));
            ++i;
        }
    }

    size_t num_phixys = lam2phixys.size();
    if (num_phixys == 0) return app;

    if (phi_lam == nullptr) {
        auto new_type = world().pi(merge_sigma(mem_lam->dom(), types), mem_lam->codom());
        phi_lam = world().nom_lam(new_type, mem_lam->dbg());
        world().DLOG("new phi_lam '{}'", phi_lam);
        world().DLOG("mem_lam => phi_lam: '{}': '{}' => '{}': '{}'", mem_lam, mem_lam->type()->dom(), phi_lam, phi_lam->dom());

        auto num_mem_vars = mem_lam->num_vars();
        size_t i = 0;
        Array<const Def*> traxy_ops(2*num_phixys + 1);
        traxy_ops[0] = phi_lam->var();
        for (auto phixy : lam2phixys) {
            traxy_ops[2*i + 1] = phixy;
            traxy_ops[2*i + 2] = phi_lam->var(num_mem_vars + i);
            ++i;
        }
        auto traxy = proxy(phi_lam->var()->type(), traxy_ops, Traxy);

        Array<const Def*> new_vars(num_mem_vars, [&](size_t i) { return traxy->out(i); });
        phi_lam->set(mem_lam->apply(world().tuple(mem_lam->dom(), new_vars)));
    } else {
        world().DLOG("reuse phi_lam '{}'", phi_lam);
    }

    auto phi = lam2phixys.begin();
    Array<const Def*> args(num_phixys, [&](auto) { return get_val(cur_nom(), *phi++); });
    return world().app(phi_lam, merge_tuple(app->arg(), args));
}

undo_t SSAConstr::analyze(const Proxy* proxy) {
    if (auto sloxy = isa_proxy(proxy, Sloxy)) {
        auto sloxy_lam = get_sloxy_lam(sloxy);

        if (keep_.emplace(sloxy).second) {
            world().DLOG("keep: '{}'; pointer needed for: '{}'", sloxy, proxy);
            return data(sloxy_lam).enter_undo;
        }
    } else if (auto phixy = isa_proxy(proxy, Phixy)) {
        auto [sloxy, mem_lam] = split_phixy(phixy);
        auto&& phixys = lam2phixys_[mem_lam];

        if (phixys.emplace(sloxy).second) {
            auto undo = data(mem_lam).visit_undo;
            assertf(undo != No_Undo, "no visit_undo for '{}'", mem_lam);
            world().DLOG("phi needed: phixy '{}' for sloxy '{}' for mem_lam '{}'", phixy, sloxy, mem_lam);
            return undo;
        }
    }

    return No_Undo;
}

undo_t SSAConstr::analyze(const Def* def) {
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto suc_lam = def->op(i)->isa_nom<Lam>(); suc_lam && !ignore(suc_lam)) {
            auto& suc_info = data(suc_lam);

            if (suc_lam->is_basicblock() && suc_lam != cur_nom()) // TODO this is a bit scruffy - maybe we can do better
                suc_info.writable.insert_range(range(data(cur_nom()).writable));

            if (!isa_callee(def, i)) {
                if (suc_info.pred) {
                    world().DLOG("'{}' -> '{}'", cur_nom(), suc_lam);
                    suc_info.pred = nullptr;
                } else {
                    world().DLOG("several preds in non-callee position; wait for EtaExp");
                    suc_info.pred = cur_nom();
                }
            }
        }
    }
    return No_Undo;
}

}
