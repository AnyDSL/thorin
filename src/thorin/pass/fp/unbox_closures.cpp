
#include "thorin/pass/fp/unbox_closures.h"
#include "thorin/transform/closure_conv.h"

namespace thorin {

const Def* UnboxClosure::rewrite(const Def* def) { 
    auto& w = world();

    if (auto proj = def->isa<Extract>(); proj && isa_ctype(proj->type())) {
        auto tuple = proj->tuple()->isa<Tuple>();
        if (!tuple || tuple->num_ops() <= 0)
            return def;
        DefVec envs, lams;
        const Def* fnc_type = nullptr;
        for (auto op: tuple->ops()) {
            auto c = isa_closure(op);
            // TODO: We have to check if the pi's and not just the environmen-types are *equal*, since
            // extract doesn't check for equiv and the closure conv may rewrite noms with different, but equiv noms
            if (!c || !c.fnc_as_lam() || (fnc_type && fnc_type != c.fnc_type()))
                return def;
            fnc_type = c.fnc_type();
            envs.push_back(c.env());
            lams.push_back(c.fnc_as_lam());
        }
        auto t = w.tuple(envs);
        auto l = w.tuple(lams);
        auto env = w.extract(w.tuple(envs), proj->index());
        auto lam = w.extract(w.tuple(lams), proj->index());
        auto new_def = w.tuple(proj->type(), {env, lam});
        w.DLOG("flattend branch: {} => {}", tuple, new_def);
        return new_def;
    }

    if (auto app = def->isa<App>()) {
        auto bxd_lam = app->callee()->isa_nom<Lam>();
        if (ignore(bxd_lam) || keep_.contains(bxd_lam))
            return def;
        auto& arg_spec = data(bxd_lam);
        DefVec doms, args, proxy_ops = {bxd_lam};
        for (size_t i = 0; i < app->num_args(); i++) {
            auto arg = app->arg(i);
            auto type = arg->type();
            if (!isa_ctype(type) || keep_.contains(bxd_lam->var(i))) {
                doms.push_back(type);
                args.push_back(arg);
                continue;
            }
            auto c = isa_closure(arg);
            if (!c) {
                w.DLOG("{},{} => ⊤ (no closure lit)" , bxd_lam, i);
                keep_.emplace(bxd_lam->var(i));
                proxy_ops.push_back(arg);
                continue;
            }
            if (arg_spec[i] && arg_spec[i] != c.env_type()) {
                w.DLOG("{},{}: {} => ⊤  (env mismatch: {})" , bxd_lam, i, arg_spec[i], c.env_type());
                keep_.emplace(bxd_lam->var(i));
                proxy_ops.push_back(arg);
                continue;
            }
            if (!arg_spec[i]) {
                arg_spec[i] = c.env_type();
                w.DLOG("{}, {}: ⊥ => {}", bxd_lam, i, c.env_type());
            }
            doms.push_back(c.env_type());
            doms.push_back(c.fnc_type());
            args.push_back(c.env());
            args.push_back(c.fnc());
        }

        if (proxy_ops.size() > 1) {
            return proxy(def->type(), proxy_ops);
        }

        auto& [ubxd_lam, old_doms] = boxed2unboxed_[bxd_lam];
        if (!ubxd_lam || old_doms != doms) {
            old_doms = doms;
            ubxd_lam = bxd_lam->stub(w, w.cn(w.sigma(doms)), bxd_lam->dbg());
            ubxd_lam->set_name(bxd_lam->name());
            size_t j = 0;
            auto new_args = w.tuple(DefArray(bxd_lam->num_doms(), [&](auto i) {
                if (auto ct = isa_ctype(bxd_lam->dom(i)); ct && !keep_.contains(bxd_lam->var(i)))
                    return w.tuple(ct, {ubxd_lam->var(j++), ubxd_lam->var(j++)});
                else
                    return ubxd_lam->var(j++);
            }));
            ubxd_lam->set(bxd_lam->apply(new_args));
            w.DLOG("replaced lam {} => {}", bxd_lam, ubxd_lam);
            keep_.insert(ubxd_lam);
        }
        return w.app(ubxd_lam, args);
    }
    return def;
}

undo_t UnboxClosure::analyze(const Proxy* def) {
    auto lam = def->op(0_u64)->isa_nom<Lam>();
    assert(lam);
    world().DLOG("undo {}", lam);
    return undo_visit(lam);
}

};

