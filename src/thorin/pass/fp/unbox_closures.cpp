
#include "thorin/pass/fp/unbox_closures.h"
#include "thorin/transform/closure_conv.h"

namespace thorin {

// static const Def* top(World& w) {
//     return w.top(w.kind());
// }

// UnboxClosure::Res UnboxClosure::unbox(const Def* def) {
//     auto& w = world();
//     if (auto c = isa_closure(def)) {
//         return {c.env()->type(), c.env(), c.lam()};
//     } else if (auto proj = def->isa<Extract>()) {
//         auto [type, env, lam] = unbox(proj->tuple());
//         if (type == top(w))
//             return {type, nullptr, nullptr};
//         auto idx = proj->index();
//         return {type, w.extract(env, idx), w.extract(lam, idx)};
//     } else if (auto tuple = def->isa<Tuple>()) {
//         assert (tuple->num_ops() > 0 && "empty tuple in closure expr");
//         auto [type, env0, lam0] = unbox(tuple->op(0));
//         DefVec envs = {env0}, lams = {lam0};
//         for (size_t i = 1; i < tuple->num_ops(); i++) {
//             auto [other_type, env, lam] = unbox(def->op(i));
//             if (type != other_type)
//                 return {};
//             envs.push_back(env);
//             lams.push_back(lam);
//         }
//         return {type, w.tuple(envs), w.tuple(lams)};
//     } else {
//         return {top(w), nullptr, nullptr};
//     }
// }

const Def* UnboxClosure::rewrite(const Def* def) { 
    auto& w = world();

    if (auto proj = def->isa<Extract>(); proj && isa_ctype(proj->type())) {
        auto tuple = proj->tuple()->isa<Tuple>();
        if (!tuple || tuple->num_ops() <= 0)
            return def;
        DefVec envs, lams;
        const Def* env_type = nullptr;
        for (auto op: tuple->ops()) {
            auto c = isa_closure(op);
            if (!c || !c.fnc_as_lam() || (env_type && !checker_.equiv(env_type, c.env_type())))
                return def;
            env_type = c.env_type();
            envs.push_back(c.env());
            lams.push_back(c.fnc_as_lam());
        }
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
            if (arg_spec[i] && !checker_.equiv(arg_spec[i], c.env_type())) {
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

