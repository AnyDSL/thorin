#ifndef THORIN_TRANSFORM_REWRITE_H
#define THORIN_TRANSFORM_REWRITE_H

#include "thorin/def.h"

namespace thorin {

class Scope;

class Rewriter {
public:
    Rewriter(World& old_world, World& new_world, const Scope* scope = nullptr);

    const Def* rewrite(const Def*);

    World& old_world;
    World& new_world;
    const Scope* scope;
    Def2Def old2new;
};

void cleanup(World&);
const Def* drop(Lam* lam, const Def* arg);

}

#endif
