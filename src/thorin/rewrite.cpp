#include "thorin/rewrite.h"

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

const Def* Rewriter::rewrite(const Def* old_def) {
    if (auto new_def = old2new.lookup(old_def)) return *new_def;
    if (scope != nullptr && !scope->contains(old_def)) return old_def;

    auto new_type = rewrite(old_def->type());
    auto new_dbg = old_def->debug() ? rewrite(old_def->debug()) : nullptr;

    if (auto old_nom = old_def->isa_nominal()) {
        auto new_nom = old_nom->stub(new_world, new_type, new_dbg);
        old2new[old_nom] = new_nom;

        for (size_t i = 0, e = old_nom->num_ops(); i != e; ++i) {
            if (auto old_op = old_nom->op(i)) new_nom->set(i, rewrite(old_op));
        }

        if (auto new_def = new_nom->restructure()) return old2new[old_nom] = new_def;

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

const Def* rewrite(Def* nom, const Def* arg, size_t i, const Scope& scope) {
    return rewrite(nom->op(i), nom->param(), arg, scope);
}

const Def* rewrite(Def* nom, const Def* arg, size_t i) {
    Scope scope(nom);
    return rewrite(nom, arg, i, scope);
}

Array<const Def*> rewrite(Def* nom, const Def* arg, const Scope& scope) {
    Rewriter rewriter(nom->world(), &scope);
    rewriter.old2new[nom->param()] = arg;
    return Array<const Def*>(nom->num_ops(), [&](size_t i) { return rewriter.rewrite(nom->op(i)); });
}

Array<const Def*> rewrite(Def* nom, const Def* arg) {
    Scope scope(nom);
    return rewrite(nom, arg, scope);
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
