#ifndef ANYDSL_AIR_JUMP_H
#define ANYDSL_AIR_JUMP_H

#include "anydsl/air/airnode.h"
#include "anydsl/air/lambda.h"
#include "anydsl/air/use.h"

namespace anydsl {

class Jump : public AIRNode {
private:

    /// Do not copy-create a \p Jump instance.
    Jump(const Jump&);
    /// Do not copy-assign a \p Jump instance.
    Jump& operator = (const Jump&);

public:

    /**
     * Construct a Jump from a pointer to the embedding Terminator 
     * and the target Lambda.
     * The parameter \p parent is needed 
     * in order to pass it to the constructor of Args.
     * Args in turn needs it in order to automatically equip appended \p Use%s
     * (arguments) with this parent.
     * Let \p parent point to the Terminator where this Jump is embedded.
     */
    Jump(Lambda* parent, Def* to)
        : AIRNode(Index_Jump)
        , to(this, to)
        , args(this)
    {}

    Lambda* toLambda() { return to.def()->isa<Lambda>(); }
    const Lambda* toLambda() const { return to.def()->isa<Lambda>(); }

    Lambda* parent() { return parent_; }
    const Lambda* parent() const { return parent_; }

    Use to;
    Args args;

    World& world() { return to.world(); }

private:

    Lambda* parent_;

    friend class World;
};

} // namespace anydsl

#endif // ANYDSL_AIR_JUMP_H
