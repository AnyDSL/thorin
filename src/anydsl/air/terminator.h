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

    Terminator();
    ~Terminator();
};

//------------------------------------------------------------------------------

/// Helper class for \p Terminator%s.
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
        : to(to, parent, debug)
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

    Jump jump;
};


//------------------------------------------------------------------------------

class Branch : public Terminator {
public:

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

    Use fct_;
    Args args;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TERMINATOR_H
