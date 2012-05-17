#ifndef ANYDSL_AIR_TERMINATOR_H
#define ANYDSL_AIR_TERMINATOR_H

#include <boost/array.hpp>

#include "anydsl/air/airnode.h"
#include "anydsl/air/constant.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Terminator : public AIRNode {
public:

    Terminator();
    ~Terminator();
};

//------------------------------------------------------------------------------

class Jump {
private:

    /// Do not create "default" \p Jump instances
    Jump();
    /// Do not copy-create a \p Jump instance.
    Jump(const Jump&);
    /// Do not copy-assign a \p Jump instance.
    Jump& operator = (const Jump&);


public:

    Jump(Terminator* parent, Lambda* to, const std::string& debug)
        : to_(to, parent, debug)
        , args(parent)
    {}


    const Use& to() const { return to_; }
    const Lambda* toLambda() const { return to_.def()->as<Lambda>(); }

private:

    Use to_;

public:

    Args args;
};

//------------------------------------------------------------------------------

class Goto : public Terminator {
public:

    const Jump& jump() const { return jump_; }

private:

    Jump jump_;
};


//------------------------------------------------------------------------------

class Branch : public Terminator {
public:
    typedef boost::array<Jump*, 2> TFJump;
    typedef boost::array<const Jump*, 2> ConstTFJump;

    const Use& cond() const { return cond_; }
    TFJump tfjump() { return (TFJump){{ &tjump, &fjump }}; }
    ConstTFJump tfjump() const { return (ConstTFJump){{ &tjump, &fjump }}; }

private:

    Use cond_;

public:

    Jump tjump;
    Jump fjump;
};

//------------------------------------------------------------------------------

class Invoke : public Terminator {
private:

public:

    const Use& fct() const { return fct_; }

private:

    Use fct_;

public:

    Args args;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
