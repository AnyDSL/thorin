#include "thorin/pass/fp/ssa_constr.h"

#include "thorin/util.h"

namespace thorin {

static const Def* get_sloxy_type(const Proxy* sloxy) { return as<Tag::Ptr>(sloxy->type())->arg(0); }
static Lam* get_sloxy_lam(const Proxy* sloxy) { return sloxy->op(0)->as_nominal<Lam>(); }
static std::tuple<const Proxy*, Lam*> split_phixy(const Proxy* phixy) { return {phixy->op(0)->as<Proxy>(), phixy->op(1)->as_nominal<Lam>()}; }

void SSAConstr::enter(Def* nom) {
    if (auto lam = nom->isa<Lam>()) {
        insert<Lam2Info>(lam); // create undo point
        lam2sloxy2val_[lam].clear();
    }
}

const Def* SSAConstr::rewrite(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_lam == nullptr) return def;

    if (auto traxy = isa_proxy(def, Traxy)) {
        world().DLOG("traxy '{}'", traxy);
        for (size_t i = 1, e = def->num_ops(); i != e; i += 2)
            set_val(cur_lam, as_proxy(traxy->op(i), Sloxy), traxy->op(i+1));
        return traxy->op(0);
    } else if (auto slot = isa<Tag::Slot>(def)) {
        auto [mem, id] = slot->args<2>();
        auto [_, ptr] = slot->split<2>();
        auto sloxy = proxy(ptr->type(), {cur_lam, id}, Sloxy, slot->debug());
        world().DLOG("sloxy: '{}'", sloxy);
        if (!keep_.contains(sloxy)) {
            set_val(cur_lam, sloxy, world().bot(get_sloxy_type(sloxy)));
            auto&& [info, _, __] = insert<Lam2Info>(cur_lam);
            info.writable.emplace(sloxy);
            return world().tuple({mem, sloxy});
        }
    } else if (auto load = isa<Tag::Load>(def)) {
        auto [mem, ptr] = load->args<2>();
        if (auto sloxy = isa_proxy(ptr, Sloxy))
            return world().tuple({mem, get_val(cur_lam, sloxy)});
    } else if (auto store = isa<Tag::Store>(def)) {
        auto [mem, ptr, val] = store->args<3>();
        if (auto sloxy = isa_proxy(ptr, Sloxy)) {
            if (auto&& [info, _, __] = insert<Lam2Info>(cur_lam); info.writable.contains(sloxy)) {
                set_val(cur_lam, sloxy, val);
                return mem;
            }
        }
    } else if (auto app = def->isa<App>()) {
        if (auto mem_lam = app->callee()->isa_nominal<Lam>(); !ignore(mem_lam))
            return mem2phi(cur_lam, app, mem_lam);
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
    } else if (auto&& [info, _, __] = insert<Lam2Info>(lam); info.pred != nullptr) {
        world().DLOG("get_val recurse: '{}': '{}' -> '{}'", sloxy, info.pred, lam);
        return get_val(info.pred, sloxy);
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

const Def* SSAConstr::mem2phi(Lam* cur_lam, const App* app, Lam* mem_lam) {
    auto& lam2phixys = lam2phixys_[mem_lam];
    if (lam2phixys.empty()) return app;

    insert<Lam2Info>(mem_lam); // create undo
    auto& phi_lam = mem2phi_.emplace(mem_lam, nullptr).first->second;

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
        auto new_type = world().pi(merge_sigma(mem_lam->domain(), types), mem_lam->codomain());
        phi_lam = world().lam(new_type, mem_lam->debug());
        world().DLOG("new phi_lam '{}'", phi_lam);
        world().DLOG("mem_lam => phi_lam: '{}': '{}' => '{}': '{}'", mem_lam, mem_lam->type()->domain(), phi_lam, phi_lam->domain());
        auto [_, ins] = preds_n_.emplace(phi_lam);
        assert(ins);

        auto num_mem_params = mem_lam->num_params();
        size_t i = 0;
        Array<const Def*> traxy_ops(2*num_phixys + 1);
        traxy_ops[0] = phi_lam->param();
        for (auto phixy : lam2phixys) {
            traxy_ops[2*i + 1] = phixy;
            traxy_ops[2*i + 2] = phi_lam->param(num_mem_params + i);
            ++i;
        }
        auto traxy = proxy(phi_lam->param()->type(), traxy_ops, Traxy);

        Array<const Def*> new_params(num_mem_params, [&](size_t i) { return traxy->out(i); });
        phi_lam->set(mem_lam->apply(world().tuple(mem_lam->domain(), new_params)));
    } else {
        world().DLOG("reuse phi_lam '{}'", phi_lam);
    }

    auto phi = lam2phixys.begin();
    Array<const Def*> args(num_phixys, [&](auto) { return get_val(cur_lam, *phi++); });
    return world().app(phi_lam, merge_tuple(app->arg(), args));
}

undo_t SSAConstr::analyze(Def* cur_nom) {
    analyzed_.clear();
    undo_t undo = No_Undo;
    for (auto op : cur_nom->extended_ops())
        undo = std::min(undo, analyze(cur_nom, op));
    return undo;
}

undo_t SSAConstr::analyze(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_nom == nullptr || def->is_const() || def->isa_nominal() || def->isa<Param>() || !analyzed_.emplace(def).second) return No_Undo;
    if (auto proxy = def->isa<Proxy>(); proxy && proxy->index() != index()) return No_Undo;

    if (auto sloxy = isa_proxy(def, Sloxy)) {
        auto sloxy_lam = get_sloxy_lam(sloxy);

        if (keep_.emplace(sloxy).second) {
            world().DLOG("keep: '{}'; pointer needed for: '{}'", sloxy, def);
            auto&& [_, undo, __] = insert<Lam2Info>(sloxy_lam);
            return undo;
        }
    } else if (auto phixy = isa_proxy(def, Phixy)) {
        auto [sloxy, mem_lam] = split_phixy(phixy);
        auto& phixys = lam2phixys_[mem_lam];

        if (phixys.emplace(sloxy).second) {
            auto&& [_, undo, __] = insert<Lam2Info>(mem_lam);
            world().DLOG("phi needed: phixy '{}' for sloxy '{}' for mem_lam '{}' -> state {}", phixy, sloxy, mem_lam, undo);
            return undo;
        }
    } else {
        auto undo = No_Undo;
        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            undo = std::min(undo, analyze(cur_nom, def->op(i)));

            if (auto suc_lam = def->op(i)->isa_nominal<Lam>(); suc_lam != nullptr && !ignore(suc_lam)) {
                auto&& [suc_info, u, ins] = insert<Lam2Info>(suc_lam);

                if (suc_lam->is_basicblock() && suc_lam != cur_lam) {
                    // TODO this is a bit scruffy - maybe we can do better
                    auto&& [cur_info, _, __] = insert<Lam2Info>(cur_lam);
                    suc_info.writable.insert_range(range(cur_info.writable));
                }

                if (!preds_n_.contains(suc_lam)) {
                    if (ins) {
                        world().DLOG("bot -> preds_1: '{}' <- pred '{}'", suc_lam, cur_lam);
                        suc_info.pred = cur_lam;
                    } else {
                        preds_n_.emplace(suc_lam);
                        world().DLOG("preds_1 -> preds_n: '{}'", suc_lam);
                        undo = std::min(undo, u);
                    }
                }
            }
        }

        return undo;
    }

    return No_Undo;
}

}
