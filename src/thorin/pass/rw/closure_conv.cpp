#include "thorin/pass/rw/closure_conv.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* ClosureConv::rewrite(Def* cur_nom, const Def* def) {
    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_lam == nullptr) return def;

    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        if (is_callee(def, i)) continue;

        if (auto lam = def->op(i)->isa_nominal<Lam>()) {
            convert(lam)->dump(2);
        }
    }

    return def;
}

const Sigma* ClosureConv::convert(const Pi* pi) {
    auto [i, ins] = pi2closure_.emplace(pi, nullptr);
    if (!ins) return i->second;

    // [Env: *, env: Env, [...., Env] -> codomain]
    auto closure = world().nom_sigma(3);
    closure->set(0, world().kind());
    auto Env = closure->param(0_s, world().dbg("Env"));
    closure->set(1, Env);
    closure->set(2, world().pi(merge_sigma(pi->domain(), {Env}), pi->codomain()));

    return i->second = closure;
}

const Tuple* ClosureConv::convert(Lam* lam) {
    auto [it, ins] = lam2closure_.emplace(lam, nullptr);
    if (!ins) return it->second;

    auto Closure = convert(lam->type());

    // [Env: *, env: Env, [...., Env] -> codomain]
    Scope scope(lam);
    const auto& free = scope.free();
    size_t n = free.size();
    Array<const Def*> Env(n);
    Array<const Def*> env(n);

    size_t i = 0;
    for (auto def : free) {
        def->dump(0);
        Env[i] = def->type();
        env[i] = def;
        ++i;
    }

    auto pi = lam->type();
    auto new_domain = world().pi(merge_sigma(pi->domain(), {world().sigma(Env)}), pi->codomain());
    new_domain->dump(1);
    auto new_lam = world().nom_lam(new_domain, lam->dbg());

    Rewriter rewriter(world(), &scope);
    i = 0;
    for (auto def : free)
        rewriter.old2new[def] = world().extract(new_lam->params().back(), n, i++);

    new_lam->set_filter(rewriter.rewrite(lam->filter()));
    new_lam->set_body  (rewriter.rewrite(lam->body  ()));

    auto closure = world().tuple(Closure, {world().sigma(Env), new_lam, world().tuple(env)})->as<Tuple>();

    return it->second = closure;
}

}
