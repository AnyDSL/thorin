#ifndef ANYDSL_AIR_TERMINATOR_H
#define ANYDSL_AIR_TERMINATOR_H

#include <boost/array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "anydsl/air/airnode.h"
#include "anydsl/air/lambda.h"
#include "anydsl/air/use.h"

// No, this file is not about this guy:
// http://en.wikipedia.org/wiki/The_Terminator

namespace anydsl {

class Lambda;

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

class DirectJump;

/**
 * Helper class for \p Terminator%s.
 *
 * This class is supposed to be embedded in other \p Terminator%s.
 * \p Jump already has enough encapsulation magic. 
 * No need to hammer further getters/setters around a Jump aggregate within a class.
 * Just make it a public class member.
 */
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
    Jump(Terminator* parent, Def* to)
        : to(parent, to)
        , args(this)
    {}

    Terminator* parent() { return parent_; }
    const Terminator* parent() const { return parent_; }

    // Jump/DirectJump do not have vtables so we test 'to'
    DirectJump* asDirectJump() { 
        anydsl_assert(to.def()->isa<Lambda>(), "not a direct jump");
        return (DirectJump*) this;
    }

    DirectJump* isaDirectJump() { return to.def()->isa<Lambda>() ? (DirectJump*) this : 0; }

    Use to;
    Args args;

private:

    Terminator* parent_;

    friend class World;
};

class DirectJump : public Jump {
public:

    DirectJump(Terminator* parent, Lambda* to) : Jump(parent, to) {}

    Lambda* lambda() { return to.def()->as<Lambda>(); }
    const Lambda* lambda() const { return to.def()->as<Lambda>(); }

};

//------------------------------------------------------------------------------

class Goto : public Terminator {
private:

    Goto(Lambda* parent, Lambda* to)
        : Terminator(parent, Index_Goto)
        , jump(this, to)
    {}

public:

    DirectJump jump;

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

    typedef boost::array<DirectJump*, 2> TFJump;
    typedef boost::array<const DirectJump*, 2> ConstTFJump;

    TFJump tfjump() { return (TFJump){{ &tjump, &fjump }}; }
    ConstTFJump tfjump() const { return (ConstTFJump){{ &tjump, &fjump }}; }

    Use cond;
    DirectJump tjump;
    DirectJump fjump;

    friend class World;
};

//------------------------------------------------------------------------------

class Invoke : public Terminator {
private:

    Invoke(Lambda* parent, Def* to)
        : Terminator(parent, Index_Invoke)
        , jump(this, to)
    {}

public:

    Jump jump;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
