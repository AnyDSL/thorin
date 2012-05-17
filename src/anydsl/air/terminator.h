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
public:

    Jump(Terminator* parent, Lambda* to, const std::string& debug)
        : to_(to, parent, debug)
        , args_(parent)
    {}


    const Use& to() const { return to_; }
    const Lambda* toLambda() const { return to_.def()->as<Lambda>(); }
    Args& args() { return args_; }
    const Args& args() const { return args_; }

private:

    Use to_;
    Args args_;
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
    typedef boost::array<const Jump*, 2> JumpTF;

    const Use& cond() const { return cond_; }
    const Jump& jumpT() const { return jumpT_; }
    const Jump& jumpF() const { return jumpF_; }
    JumpTF jumpTF() const { return (JumpTF){{ &jumpT_, &jumpF_ }}; }

private:

    Use cond_;
    Jump jumpT_;
    Jump jumpF_;
};

//------------------------------------------------------------------------------

class Invoke : public Terminator {
private:

public:

    const Use& fct() const { return fct_; }
    const Args& args() const { return args_; }

private:

    Use fct_;
    Args args_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
