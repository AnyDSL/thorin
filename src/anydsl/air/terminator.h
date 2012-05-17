#ifndef ANYDSL_AIR_TERMINATOR_H
#define ANYDSL_AIR_TERMINATOR_H

#include <list>

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
    const Lambda* toLambda() const { return scast<Lambda>(to_.def()); }
    Args& args() { return args_; }

private:

    Use to_;
    Args args_;
};

//------------------------------------------------------------------------------

class Branch : public Terminator {
public:

    const Use& cond() const { return cond_; }
    const Use& useT() const { return useT_; }
    const Use& useF() const { return useF_; }
    const Lambda* lambdaT() const { return scast<Lambda>(useT_.def()); }
    const Lambda* lambdaF() const { return scast<Lambda>(useF_.def()); }

private:

    Use cond_;
    Use useT_;
    Use useF_;
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
