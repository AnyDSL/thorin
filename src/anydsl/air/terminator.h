#ifndef ANYDSL_AIR_TERMINATOR_H
#define ANYDSL_AIR_TERMINATOR_H

#include <list>

#include "anydsl/air/airnode.h"
#include "anydsl/air/use.h"

namespace anydsl {

class Lambda;

class Terminator : public AIRNode {
public:

    Terminator();
    ~Terminator();
};

//------------------------------------------------------------------------------

class Branch : public Terminator {
public:

    const Use& cond() const { return cond_; }
    const Lambda* ifTrue()  { return ifTrue_; }
    const Lambda* ifFalse() { return ifFalse_; }

private:

    Use cond_;
    Lambda* ifTrue_;
    Lambda* ifFalse_;
};

//------------------------------------------------------------------------------

typedef std::list<Use> Uses;

class Invoke : public Terminator {
public:

    const Use& fct() const { return fct_; }
    const Uses& uses() const { return uses_; }

private:

    Use fct_;
    Uses uses_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
