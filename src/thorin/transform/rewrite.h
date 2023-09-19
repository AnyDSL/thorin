#ifndef THORIN_REWRITE_H
#define THORIN_REWRITE_H

#include "thorin/world.h"

namespace thorin {

class Rewriter {
public:
    explicit Rewriter(World& src, World& dst);
    explicit Rewriter(World& world) : Rewriter(world, world) {}

    const Def* instantiate(const Def* odef);
    const Def* insert(const Def* odef, const Def* ndef);

    World& src() { return src_; }
    World& dst() { return dst_; }

protected:
    explicit Rewriter(World& src, World& dst, Rewriter& parent);
    virtual const Def* lookup(const Def* odef);
    virtual const Def* rewrite(const Def* odef);

private:
    Def2Def old2new_;
    World& src_;
    World& dst_;
};

}

#endif