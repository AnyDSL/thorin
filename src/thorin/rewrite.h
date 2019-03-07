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
