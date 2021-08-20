#include "thorin/pass/rw/closure_conv.h"

#include "thorin/rewrite.h"

namespace thorin {

const Def* ClosureConv::rewrite(Def* old_nom, const Def*, const Def* ) {
#if 0
    if (old_nom->type() != new_type) {
        auto new_nom = old_nom->stub(world(), new_type, new_dbg);
        new_nom->set(old_nom->apply(proxy(old_nom->var()->type(), {new_nom->var()}, 0)));

        if (old_nom->is_external()) {
            old_nom->make_internal();
            new_nom->make_external();
        }

        return new_nom;
    }

#endif
    return old_nom;
}

const Def* ClosureConv::rewrite(const Def* def) {
    auto cur_lam = cur_nom<Lam>();
    if (cur_lam == nullptr || def->isa<Var>()) return def;

    size_t n = def->num_ops();
    Array<const Def*> new_ops(n);

    for (size_t i = 0; i != n; ++i) {
        auto op = def->op(i);
        if (auto lam = op->isa_nom<Lam>())
            new_ops[i] = close(lam);
        else if (auto pi = op->isa<Pi>())
            new_ops[i] = close(pi);
        else
            new_ops[i] = op;
    }

    if (auto app = def->isa<App>()) {
        auto [Env, env, f] = new_ops[0]->split<3>();
        new_ops[1] = merge_tuple(new_ops[1], {env});
        new_ops[0] = f;
    }

    return def->rebuild(world(), def->type(), new_ops, def->dbg());
}

const Sigma* ClosureConv::close(const Pi* pi) {
    auto [i, ins] = pi2closure_.emplace(pi, nullptr);
    if (!ins) return i->second;

    // A -> B  =>  [Env: *, env: Env, [A ∘ Env] -> B]
    auto closure = world().nom_sigma(3);
    closure->set(0, world().kind());
    auto Env = closure->var(0_s, world().dbg("Env"));
    closure->set(1, Env);
    auto new_dom = merge_sigma(pi->dom(), {Env});
    auto new_pi = world().pi(new_dom, pi->codom());
    closure->set(2, new_pi);

    return i->second = closure;
}

const Tuple* ClosureConv::close(Lam* lam) {
    auto [it, ins] = lam2closure_.emplace(lam, nullptr);
    if (!ins) return it->second;

    auto Closure = close(lam->type());

    Scope scope(lam);
    const auto& free = scope.free_defs();
    size_t n = free.size();
    Array<const Def*> Envs(n);
    Array<const Def*> envs(n);

    n = 0;
    for (auto def : free) {
        if (def->has_dep(Dep::Var)) {
            outf("free: {}/{}\n", n, def);
            def->dump(1);
            Envs[n] = def->type();
            envs[n] = def;
            ++n;
        }
    }

    Envs.shrink(n);
    envs.shrink(n);

    auto Env = world().sigma(Envs);
    auto env = world().tuple(envs);

    auto pi = lam->type();
    auto new_dom = merge_sigma(pi->dom(), {Env});
    auto new_pi  = world().pi(new_dom, pi->codom());
    auto new_lam = world().nom_lam(new_pi, lam->dbg());

    Rewriter rewriter(world(), &scope);
    rewriter.old2new[lam->var()] = world().tuple(new_lam->vars().skip_back());

    for (size_t i = 0; i != n; ++i)
        rewriter.old2new[envs[i]] = new_lam->vars().back()->out(n, i);

    new_lam->set_filter(rewriter.rewrite(lam->filter()));
    new_lam->set_body  (rewriter.rewrite(lam->body  ()));

    auto closure = world().tuple(Closure, {Env, env, new_lam})->as<Tuple>();

    return it->second = closure;
}

}
