#ifndef ANYDSL_AIR_JUMP_H
#define ANYDSL_AIR_JUMP_H

#include "anydsl/air/airnode.h"
#include "anydsl/air/lambda.h"
#include "anydsl/air/use.h"

namespace anydsl {

class Jump : public Def {
public:

    /**
     * Construct a Jump from a pointer to the embedding Terminator 
     * and the target Lambda.
     * The parameter \p parent is needed 
     * in order to pass it to the constructor of Ops.
     * Args in turn needs it in order to automatically equip appended \p Use%s
     * (arguments) with this parent.
     * Let \p parent point to the Terminator where this Jump is embedded.
     */
    Jump(Lambda* parent, Def* to);

    Lambda* toLambda() { return ccast<Lambda>(to.def()->isa<Lambda>()); }
    const Lambda* toLambda() const { return to.def()->isa<Lambda>(); }

    Lambda* parent() { return parent_; }
    const Lambda* parent() const { return parent_; }

    const Use& to;

    Ops& ops() { return ops_; }

private:

    Lambda* parent_;

    friend class World;
};

} // namespace anydsl

#endif // ANYDSL_AIR_JUMP_H
