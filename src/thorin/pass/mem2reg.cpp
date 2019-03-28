#include "thorin/pass/mem2reg.h"

#include "thorin/util.h"
#include "thorin/util/log.h"

namespace thorin {

static const Def* proxy_type(const Analyze* proxy) { return proxy->type()->as<PtrType>()->pointee(); }

const Analyze* Mem2Reg::isa_proxy(const Def* ptr) {
    if (auto analyze = ptr->isa<Analyze>(); analyze && analyze->index() == index() && !analyze->op(1)->isa<Analyze>()) return analyze;
    return nullptr;
}

const Def* Mem2Reg::rewrite(const Def* def) {
    if (auto slot = def->isa<Slot>()) {
        auto proxy = world().analyze(slot->out_ptr_type(), {man().cur_lam(), world().lit_nat(lam2info(man().cur_lam()).num_slots++)}, index(), slot->debug());
        auto& info = proxy2info(proxy, man().cur_state_id());
        outf("slot: {}\n", proxy);
        if (info.lattice == ProxyInfo::SSA) {
            man().new_state();
            set_val(proxy, world().bot(proxy_type(proxy)));
            auto& info = lam2info(man().cur_lam());
            info.writable.emplace(proxy);
            return world().tuple({slot->mem(), proxy});
        }
    } else if (auto load = def->isa<Load>()) {
        if (auto proxy = isa_proxy(load->ptr()))
            return world().tuple({load->mem(), get_val(proxy)});
    } else if (auto store = def->isa<Store>()) {
        if (auto proxy = isa_proxy(store->ptr())) {
            if (lam2info(man().cur_lam()).writable.contains(proxy)) {
                set_val(proxy, store->val());
                return store->mem();
            }
        }
    } else if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            const auto& info = lam2info(lam);
            if (auto new_lam = info.new_lam) {
                outf("{}/{}:\n", lam2info(lam).undo, lam2info(new_lam).undo);
                Array<const Def*> args(info.proxies.size(), [&](auto i) { return get_val(info.proxies[i]); });
                return world().app(new_lam, merge_tuple(app->arg(), args));
            }
        }
    }

    return def;
}

void Mem2Reg::inspect(Def* def) {
    if (auto old_lam = def->isa<Lam>()) {
        auto& info = lam2info(old_lam);
        if (old_lam->is_external() || old_lam->intrinsic() != Intrinsic::None) {
            info.lattice = LamInfo::Keep;
        } else if (info.lattice != LamInfo::Keep) {
            man().new_state();

            if (info.lattice == LamInfo::PredsN && !info.proxies.empty()) {
                assert(old_lam->mem_param());
                Array<const Def*> types(info.proxies.size(), [&](auto i) { return proxy_type(info.proxies[i]); });
                auto new_domain = merge_sigma(old_lam->domain(), types);
                auto new_lam = world().lam(world().pi(new_domain, old_lam->codomain()), old_lam->debug());
                outf("new_lam: {} -> {}\n", old_lam, new_lam);
                new2old(new_lam) = old_lam;
                lam2info(new_lam).lattice = LamInfo::PredsN;
                info.new_lam = new_lam;
            }
        }
    }
}

void Mem2Reg::enter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        outf("enter: {}\n", new_lam);
        auto& info = lam2info(new_lam);
        info.proxy2val.clear();
        if (auto old_lam = new2old(new_lam)) {
            outf("enter: {}/{}\n", old_lam, new_lam);
            auto& proxies = lam2info(old_lam).proxies;
            size_t n = new_lam->num_params() - proxies.size();

            auto new_param = world().tuple(Array<const Def*>(n, [&](auto i) { return new_lam->param(i); }));
            man().map(old_lam->param(), new_param);
            new_lam->set(old_lam->ops());

            if (auto old_lam = new2old(new_lam)) {
                outf("enter: {}/{}\n", old_lam, new_lam);
                auto& proxies = lam2info(old_lam).proxies;
                size_t n = new_lam->num_params() - proxies.size();

                for (size_t i = 0, e = proxies.size(); i != e; ++i) {
                    auto proxy = proxies[i];
                    //auto lam = proxy->op(0)->as_nominal<Lam>();
                    set_val(new_lam, proxy, new_lam->param(n + i));
                }
            }
        }
    }
}

void Mem2Reg::reenter(Def* def) {
    if (auto new_lam = def->isa<Lam>()) {
        outf("enter: {}\n", new_lam);
        // remove any potential garbage from previous runs
        auto& info = lam2info(new_lam);
        info.num_slots = 0;
        //if (auto old_lam = new2old(new_lam)) {
            //outf("enter: {}/{}\n", old_lam, new_lam);
            //auto& proxies = lam2info(old_lam).proxies;
            //size_t n = new_lam->num_params() - proxies.size();

            //for (size_t i = 0, e = proxies.size(); i != e; ++i) {
                //auto proxy = proxies[i];
                ////auto lam = proxy->op(0)->as_nominal<Lam>();
                //set_val(new_lam, proxy, new_lam->param(n + i));
            //}
        //}
    }
}

const Def* Mem2Reg::get_val(Lam* lam, const Analyze* proxy) {
    const auto& info = lam2info(lam);
    if (auto val = info.proxy2val.lookup(proxy)) {
        outf("get_val {} for {}: {}\n", lam, proxy, *val);
        return *val;
    }

    switch (info.lattice) {
        case LamInfo::Preds0: return world().bot(proxy_type(proxy));
        case LamInfo::Preds1: return get_val(info.pred, proxy);
        default: {
            auto old_lam = original(lam);
            outf("virtual phi: {}/{} for {}\n", old_lam, lam, proxy);
            return set_val(lam, proxy, world().analyze(proxy_type(proxy), {old_lam, proxy}, index()));
        }
    }
}

const Def* Mem2Reg::set_val(Lam* lam, const Analyze* proxy, const Def* val) {
    outf("set_val {} for {}: {}\n", lam, proxy, val);
    return lam2info(lam).proxy2val[proxy] = val;
}

void Mem2Reg::analyze(const Def* def) {
    if (def->isa<Param>()) return;

    // we need to install a phi in lam next time around
    if (auto analyze = def->isa<Analyze>(); analyze && analyze->index() == index() && analyze->op(1)->isa<Analyze>()) {
        auto lam   = analyze->op(0)->as_nominal<Lam>();
        auto proxy = analyze->op(1)->as<Analyze>();
        auto& lam_info = lam2info(lam);
        auto& proxy_info = proxy2info(proxy);
        auto& proxies = lam_info.proxies;

        if (lam_info.lattice == LamInfo::Keep) {
            if (proxy_info.lattice == ProxyInfo::SSA) {
                proxy_info.lattice = ProxyInfo::Keep;
                outf("keep: {}\n", proxy);
                if (auto i = std::find(proxies.begin(), proxies.end(), proxy); i != proxies.end()) {
                    proxies.erase(i);
                    man().undo(proxy_info.undo);
                }
            }
        } else {
            assert(lam_info.lattice == LamInfo::PredsN);
            assertf(std::find(proxies.begin(), proxies.end(), proxy) == proxies.end(), "already added proxy {} to {}", proxy, lam);
            //assert(proxy_info.undo <= lam_info.undo);
            proxies.emplace_back(proxy);
            outf("phi needed: {}\n", analyze);
            man().undo(lam_info.undo);
        }
        return;
    }

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);

        if (auto proxy = isa_proxy(op)) {
            if (auto& info = proxy2info(proxy); info.lattice == ProxyInfo::SSA) {
                outf("keep: {}\n", proxy);
                std::cout << std::flush;
                info.lattice = ProxyInfo::Keep;
                man().undo(info.undo);
            }
        } else if (auto lam = op->isa_nominal<Lam>()) {
            // TODO optimize
            if (lam->is_basicblock() && lam != man().cur_lam())
                lam2info(lam).writable.insert_range(range(lam2info(man().cur_lam()).writable));
            auto& info = lam2info(original(lam));
            auto pred = man().cur_lam();

            switch (info.lattice) {
                case LamInfo::Preds0:
                    info.lattice = LamInfo::Preds1;
                    info.pred = pred;
                    assert(info.proxies.empty());
                    break;
                case LamInfo::Preds1:
                    info.lattice = LamInfo::PredsN;
                    outf("Preds1 -> PredsN\n");
                    man().undo(info.undo);
                    break;
                default:
                    break;
            }

            // if lam does not occur as callee and has more than one pred
            if ((!def->isa<App>() || i != 0) && (info.lattice == LamInfo::PredsN )) {
                info.lattice = LamInfo::Keep;
                outf("keep: {}\n", lam);
                for (auto proxy : info.proxies) {
                    auto& proxy_info = proxy2info(proxy);
                    proxy_info.lattice = ProxyInfo::Keep;
                    info.proxies.clear();
                    man().undo(info.undo);
                    man().undo(proxy_info.undo);
                }
            }
        }
    }
}

}
