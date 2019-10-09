#include "thorin/rewrite.h"

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

const Def* Rewriter::rewrite(const Def* old_def) {
    if (auto new_def = old2new.lookup(old_def)) return *new_def;
    if (scope != nullptr && !scope->contains(old_def)) return old_def;

    if (fn) {
        if (auto new_def = fn(old_def)) return old2new[old_def] = new_def;
    }

    auto new_type = rewrite(old_def->type());
    auto new_dbg = old_def->debug();

    if (auto old_dbg = old_def->debug(); old_dbg && &new_world != &old_world)
        new_dbg = rewrite(old_dbg);

    if (auto old_nom = old_def->isa_nominal()) {
        auto new_nom = old_nom->stub(new_world, new_type, new_dbg);
        old2new[old_nom] = new_nom;

        for (size_t i = 0, e = old_nom->num_ops(); i != e; ++i) {
            if (auto old_op = old_nom->op(i))
                new_nom->set(i, rewrite(old_op));
        }

        return new_nom;
    }

    Array<const Def*> new_ops(old_def->num_ops(), [&](auto i) { return rewrite(old_def->op(i)); });
    return old2new[old_def] = old_def->rebuild(new_world, new_type, new_ops, new_dbg);
}

const Def* rewrite(const Def* def, const Def* old_def, const Def* new_def, const Scope& scope) {
    Rewriter rewriter(def->world(), &scope);
    rewriter.old2new[old_def] = new_def;
    return rewriter.rewrite(def);
}

const Def* rewrite(Def* nom, const Def* arg, const Scope& scope) {
    return rewrite(nom->ops().back(), nom->param(), arg, scope);
}

const Def* rewrite(Def* nom, const Def* arg) {
    Scope scope(nom);
    return rewrite(nom, arg, scope);
}

const Def* rewrite(Def* nom, const Scope& scope, RewriteFn fn) {
    return Rewriter(nom->world(), &scope, fn).rewrite(nom->ops().back());
}

const Def* rewrite(Def* nom, RewriteFn fn) {
    Scope scope(nom);
    return rewrite(nom, scope, fn);
}

void cleanup(World& old_world) {
    World new_world(old_world);

    Rewriter rewriter(old_world, new_world);
    rewriter.old2new.rehash(old_world.defs().capacity());

    for (const auto& [name, nom] : old_world.externals())
        rewriter.rewrite(nom)->as_nominal()->make_external();

    swap(rewriter.old_world, rewriter.new_world);
}

}
