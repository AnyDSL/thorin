#ifndef ANYDSL_AIR_TERMINATOR_H
#define ANYDSL_AIR_TERMINATOR_H

#include <list>

#include "anydsl/air/airnode.h"
#include "anydsl/air/constant.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Lambda;

//------------------------------------------------------------------------------

class Terminator : public AIRNode {
public:

    Terminator();
    ~Terminator();
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
    const Uses& args() const { return args_; }

private:

    Use fct_;
    Uses args_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
