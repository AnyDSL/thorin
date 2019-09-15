#ifndef THORIN_TRANSFORM_REWRITE_H
#define THORIN_TRANSFORM_REWRITE_H

#include "thorin/def.h"

namespace thorin {

class Scope;

/// Rewrites part of a program.
class Rewriter {
public:
    Rewriter(World& old_world, World& new_world, const Scope* scope = nullptr);
    Rewriter(World& world, const Scope* scope = nullptr)
        : Rewriter(world, world, scope)
    {}

    const Def* rewrite(const Def*);
    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { old2new.emplace(old_def, new_def); return new_def; }
    World& world() { assert(&old_world == &new_world); return old_world; }

    World& old_world;
    World& new_world;
    const Scope* scope;
    Def2Def old2new;
};

/// Rewrites @p def while mapping @p old_def to @p new_def.
const Def* rewrite(const Def* def, const Def* old_def, const Def* new_def);
/// Inlines @p lam while using the argument @p arg.
const Def* drop(Lam* lam, const Def* arg);
/// Removes unreachable and dead code by rewriting the whole program.
void cleanup(World&);

}

#endif
