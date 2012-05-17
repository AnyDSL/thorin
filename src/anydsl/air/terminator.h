#ifndef ANYDSL_AIR_TERMINATOR_H
#define ANYDSL_AIR_TERMINATOR_H

#include <boost/array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "anydsl/air/airnode.h"
#include "anydsl/air/constant.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Terminator : public AIRNode {
public:

    Terminator(Lambda* parent, IndexKind index, const std::string& debug)
        : AIRNode(index, debug)
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

    Jump(Terminator* parent, Lambda* to, const std::string& debug)
        : to(parent, to, debug)
        , args(parent)
    {}

    Lambda* lambda() { return to.def()->as<Lambda>(); }
    const Lambda* lambda() const { return to.def()->as<Lambda>(); }

    Use to;
    Args args;
};

//------------------------------------------------------------------------------

class Goto : public Terminator {
public:

    Goto(Lambda* parent, Lambda* to, const std::string toDebug, const std::string debug)
        : Terminator(parent, Index_Goto, debug)
        , jump(this, to, toDebug)
    {}

    Jump jump;
};

//------------------------------------------------------------------------------

class Branch : public Terminator {
public:

    Branch(Lambda* parent, Def* cond, Lambda* tlambda, Lambda* flambda, 
           const std::string& condDebug, 
           const std::string& tdebug, const std::string& fdebug, 
           const std::string& debug)
        : Terminator(parent, Index_Branch, debug)
        , cond(this, cond, condDebug)
        , tjump(this, tlambda, tdebug)
        , fjump(this, flambda, fdebug)
    {}

    typedef boost::array<Jump*, 2> TFJump;
    typedef boost::array<const Jump*, 2> ConstTFJump;

    TFJump tfjump() { return (TFJump){{ &tjump, &fjump }}; }
    ConstTFJump tfjump() const { return (ConstTFJump){{ &tjump, &fjump }}; }

    Use cond;
    Jump tjump;
    Jump fjump;
};

//------------------------------------------------------------------------------

class Invoke : public Terminator {
public:

    Invoke(Lambda* parent, Def* fct, const std::string& fctDebug, const std::string& debug)
        : Terminator(parent, Index_Invoke, debug)
        , fct(this, fct, fctDebug)
        , args(this)
    {}

    Use fct;
    Args args;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
