#ifndef ANYDSL_PRIMOP_H
#define ANYDSL_PRIMOP_H

#include "anydsl/air/airnode.h"
#include "anydsl/air/enums.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

class PrimOp : public AIRNode {
public:

    PrimOpKind primOpKind() const { return (PrimOpKind) indexKind_; }

protected:

    PrimOp(IndexKind indexKind, const std::string& debug)
        : AIRNode(indexKind, debug)
    {}
};

//------------------------------------------------------------------------------

class ArithOp : public PrimOp {
public:

    ArithOp(ArithOpKind arithOpKind, 
            Def* ldef, Def* rdef, 
            const std::string& ldebug = "", const std::string& rdebug = "", 
            const std::string& debug = "")
        : PrimOp((IndexKind) arithOpKind, debug)
        , luse_(ldef, this, ldebug)
        , ruse_(rdef, this, rdebug)
    {}

    ArithOpKind arithOpKind() { return (ArithOpKind) indexKind_; }
    const Use& luse() const { return luse_; }
    const Use& ruse() const { return ruse_; }

private:

    Use luse_;
    Use ruse_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_PRIMOP_H
