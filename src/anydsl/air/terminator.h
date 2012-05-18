#ifndef ANYDSL_AIR_TERMINATOR_H
#define ANYDSL_AIR_TERMINATOR_H

#include <boost/array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "anydsl/air/airnode.h"
#include "anydsl/air/literal.h"
#include "anydsl/air/use.h"

// No, this file is not about this guy:
// http://en.wikipedia.org/wiki/The_Terminator

namespace anydsl {

//------------------------------------------------------------------------------

class Terminator : public AIRNode {
protected:

    Terminator(Lambda* parent, IndexKind index)
        : AIRNode(index)
        , parent_(parent)
    {}

    Lambda* parent() { return parent_; }
    const Lambda* parent() const { return parent_; }

private:

    Lambda* parent_;
};

//------------------------------------------------------------------------------

/// Helper class for \p Terminator%s.
class Jump {
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
    Jump(Terminator* parent, Lambda* to)
        : to(parent, to)
        , args(parent)
    {}

    Lambda* lambda() { return to.def()->as<Lambda>(); }
    const Lambda* lambda() const { return to.def()->as<Lambda>(); }

    Use to;
    Args args;

    friend class World;
};

//------------------------------------------------------------------------------

class Goto : public Terminator {
private:

    Goto(Lambda* parent, Lambda* to)
        : Terminator(parent, Index_Goto)
        , jump(this, to)
    {}

public:

    Jump jump;

    friend class World;
};

//------------------------------------------------------------------------------

class Branch : public Terminator {
private:

    Branch(Lambda* parent, Def* cond, Lambda* tlambda, Lambda* flambda)
        : Terminator(parent, Index_Branch)
        , cond(this, cond)
        , tjump(this, tlambda)
        , fjump(this, flambda)
    {}

public:

    typedef boost::array<Jump*, 2> TFJump;
    typedef boost::array<const Jump*, 2> ConstTFJump;

    TFJump tfjump() { return (TFJump){{ &tjump, &fjump }}; }
    ConstTFJump tfjump() const { return (ConstTFJump){{ &tjump, &fjump }}; }

    Use cond;
    Jump tjump;
    Jump fjump;

    friend class World;
};

//------------------------------------------------------------------------------

class Invoke : public Terminator {
private:

    Invoke(Lambda* parent, Def* fct)
        : Terminator(parent, Index_Invoke)
        , fct(this, fct)
        , args(this)
    {}

public:

    Use fct;
    Args args;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
