#include "thorin/pass/fp/ssa_constr.h"

#include "thorin/pass/fp/eta_exp.h"

namespace thorin {

static const Def* get_sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static std::tuple<const Proxy*, Lam*> split_phixy(const Proxy* phixy) { return {phixy->op(0)->as<Proxy>(), phixy->op(1)->as_nom<Lam>()}; }

void SSAConstr::enter() {
    lam2sloxy2val_[curr_nom()].clear();
}

const Def* SSAConstr::rewrite(const Proxy* proxy) {
    if (proxy->flags() == Traxy) {
        world().DLOG("traxy '{}'", proxy);
        for (size_t i = 1, e = proxy->num_ops(); i != e; i += 2)
            set_val(curr_nom(), as_proxy(proxy->op(i), Sloxy), proxy->op(i+1));
        return proxy->op(0);
    }

    return proxy;
}

const Def* SSAConstr::rewrite(const Def* def) {
    if (auto slot = isa<Tag::Slot>(def)) {
        auto [mem, id] = slot->args<2>();
        auto [_, ptr] = slot->split<2>();
        auto sloxy = proxy(ptr->type(), {curr_nom(), id}, Sloxy, slot->dbg());
        world().DLOG("sloxy: '{}'", sloxy);
        if (!keep_.contains(sloxy)) {
            set_val(curr_nom(), sloxy, world().bot(get_sloxy_type(sloxy)));
            data(curr_nom()).writable.emplace(sloxy);
            return world().tuple({mem, sloxy});
        }
    } else if (auto load = isa<Tag::Load>(def)) {
        auto [mem, ptr] = load->args<2>();
        if (auto sloxy = isa_proxy(ptr, Sloxy))
            return world().tuple({mem, get_val(curr_nom(), sloxy)});
    } else if (auto store = isa<Tag::Store>(def)) {
        auto [mem, ptr, val] = store->args<3>();
        if (auto sloxy = isa_proxy(ptr, Sloxy)) {
            if (data(curr_nom()).writable.contains(sloxy)) {
                set_val(curr_nom(), sloxy, val);
#if 0
                return world().op_remem(mem, store->dbg());
#else
                return mem;
#endif
            }
        }
    } else if (auto app = def->isa<App>()) {
        if (auto mem_lam = app->callee()->isa_nom<Lam>(); !ignore(mem_lam))
            return mem2phi(app, mem_lam);
    } else {
        // TODO I'm currently not sure why we need this.
        // The eta_exp_->new2old(...) should be enough, but removing this will break reverse.impala.
        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            if (auto lam = def->op(i)->isa_nom<Lam>(); !ignore(lam)) {
                if (mem2phi_.contains(lam))
                   return def->refine(i, eta_exp_->proxy(lam));
            }
        }
    }

    return def;
}

const Def* SSAConstr::get_val(Lam* lam, const Proxy* sloxy) {
    if (auto val = lam2sloxy2val_[lam].lookup(sloxy)) {
        world().DLOG("get_val found: '{}': '{}': '{}'", sloxy, *val, lam);
        return *val;
    } else if (lam->is_external()) {
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
    auto&& sloxys = lam2sloxys_[mem_lam];
    if (sloxys.empty()) return app;

    DefVec types, phis;
    for (auto i = sloxys.begin(), e = sloxys.end(); i != e;) {
        auto sloxy = *i;
        if (keep_.contains(sloxy)) {
            i = sloxys.erase(i);
        } else {
            phis.emplace_back(sloxy);
            types.emplace_back(get_sloxy_type(sloxy));
            ++i;
        }
    }

    size_t num_phis = phis.size();
    if (num_phis == 0) return app;

    auto&& [phi_lam, old_phis] = mem2phi_[mem_lam];
    if (phi_lam == nullptr || old_phis != phis) {
        old_phis = phis;
        auto new_type = world().pi(merge_sigma(mem_lam->dom(), types), mem_lam->codom());
        phi_lam = world().nom_lam(new_type, mem_lam->dbg());
        eta_exp_->new2old(phi_lam, mem_lam);
        world().DLOG("new phi_lam '{}'", phi_lam);

        auto num_mem_vars = mem_lam->num_vars();
        size_t i = 0;
        Array<const Def*> traxy_ops(2*num_phis + 1);
        traxy_ops[0] = phi_lam->var();
        for (auto sloxy : sloxys) {
            traxy_ops[2*i + 1] = sloxy;
            traxy_ops[2*i + 2] = phi_lam->var(num_mem_vars + i);
            ++i;
        }
        auto traxy = proxy(phi_lam->var()->type(), traxy_ops, Traxy);

        Array<const Def*> new_vars(num_mem_vars, [&](size_t i) { return traxy->out(i); });
        phi_lam->set(mem_lam->apply(world().tuple(mem_lam->dom(), new_vars)));
    } else {
        world().DLOG("reuse phi_lam '{}'", phi_lam);
    }

    world().DLOG("mem_lam => phi_lam: '{}': '{}' => '{}': '{}'", mem_lam, mem_lam->type()->dom(), phi_lam, phi_lam->dom());
    auto sloxy = sloxys.begin();
    Array<const Def*> args(num_phis, [&](auto) { return get_val(curr_nom(), *sloxy++); });
    return world().app(phi_lam, merge_tuple(app->arg(), args));
}

undo_t SSAConstr::analyze(const Proxy* proxy) {
    if (proxy->flags() == Sloxy) {
        auto sloxy_lam = proxy->op(0)->as_nom<Lam>();

        if (keep_.emplace(proxy).second) {
            world().DLOG("keep: '{}'; pointer needed", proxy);
            return undo_enter(sloxy_lam);
        }
    }

    assert(proxy->flags() == Phixy);
    auto [sloxy, mem_lam] = split_phixy(proxy);
    if (lam2sloxys_[mem_lam].emplace(sloxy).second) {
        world().DLOG("phi needed: phixy '{}' for sloxy '{}' for mem_lam '{}'", proxy, sloxy, mem_lam);
        return undo_visit(mem_lam);
    }

    return No_Undo;
}

undo_t SSAConstr::analyze(const Def* def) {
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (auto succ_lam = def->op(i)->isa_nom<Lam>(); succ_lam && !ignore(succ_lam)) {
            auto& succ_info = data(succ_lam);

            // TODO this is a bit scruffy - maybe we can do better
            if (succ_lam->is_basicblock() && succ_lam != curr_nom())
                succ_info.writable.insert_range(data(curr_nom()).writable);

            if (!isa_callee(def, i)) {
                if (succ_info.pred) {
                    world().DLOG("several preds in non-callee position; wait for EtaExp");
                    succ_info.pred = nullptr;
                } else {
                    world().DLOG("'{}' -> '{}'", curr_nom(), succ_lam);
                    succ_info.pred = curr_nom();
                }
            }
        }
    }

    return No_Undo;
}

}
