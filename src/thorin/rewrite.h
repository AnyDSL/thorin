#ifndef THORIN_TRANSFORM_REWRITE_H
#define THORIN_TRANSFORM_REWRITE_H

#include "thorin/def.h"

namespace thorin {

class Scope;

class Rewriter {
public:
    Rewriter(World& old_world, World& new_world, const Scope* scope = nullptr);
    Rewriter(World& world, const Scope* scope = nullptr)
        : Rewriter(world, world, scope)
    {}

    const Def* rewrite(const Def*);
    auto new_ops(const Def* old_def) { return Array<const Def*>(old_def->num_ops(), [&](auto i) { return rewrite(old_def->op(i)); }); }
    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { old2new.emplace(old_def, new_def); return new_def; }

    World& old_world;
    World& new_world;
    const Scope* scope;
    Def2Def old2new;
};

const Def* rewrite(const Def* def, const Def* old_def, const Def* new_def);
const Def* drop(Lam* lam, const Def* arg);
void cleanup(World&);

}

#endif
