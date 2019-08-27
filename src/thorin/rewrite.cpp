#include "thorin/rewrite.h"

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Rewriter::Rewriter(World& old_world, World& new_world, const Scope* scope)
    : old_world(old_world)
    , new_world(new_world)
    , scope(scope)
{
    old2new[old_world.end_scope()] = new_world.end_scope();
    old2new[old_world.universe()]  = new_world.universe();
}

const Def* Rewriter::rewrite(const Def* old_def) {
    if (auto new_def = old2new.lookup(old_def)) return *new_def;
    if (scope != nullptr && (!scope->contains(old_def) || scope->entry() == old_def)) return old_def;
    // HACK the entry really shouldn't be part of the scope ^^^

    auto new_type  = rewrite(old_def->type());
    auto new_debug = old_def->debug() ? rewrite(old_def->debug()) : nullptr;

    if (auto old_nom = old_def->isa_nominal()) {
        auto new_nom = old_nom->stub(new_world, new_type, new_debug);
        map(old_nom, new_nom);
        if (old_nom->is_set()) new_nom->set(new_ops(old_nom));
        return new_nom;
    }

    return map(old_def, old_def->rebuild(new_world, new_type, new_ops(old_def), new_debug));
}

const Def* rewrite(const Def* def, const Def* old_def, const Def* new_def) {
    Rewriter rewriter(def->world());
    rewriter.map(old_def, new_def);
    return rewriter.rewrite(def);
}

const Def* drop(Lam* lam, const Def* arg) {
    Scope scope(lam);
    Rewriter rewriter(lam->world(), &scope);
    rewriter.map(lam->param(), arg);
    return rewriter.rewrite(lam->body());
}

void cleanup(World& old_world) {
    World new_world(old_world);

    Rewriter rewriter(old_world, new_world);
    rewriter.old2new.rehash(old_world.defs().capacity());

    for (auto old_lam : old_world.externals())
        rewriter.rewrite(old_lam)->as_nominal<Lam>()->make_external();

    swap(rewriter.old_world, rewriter.new_world);
}

}
