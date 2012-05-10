#ifndef ANYDSL_PRIMOP_H
#define ANYDSL_PRIMOP_H

#include "anydsl/air/airnode.h"
#include "anydsl/air/enums.h"
#include "anydsl/air/def.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

class PrimOp : public Param {
public:

    PrimOpKind primOpKind() const { return (PrimOpKind) indexKind(); }

    bool compare(PrimOp* other) const;

protected:

    PrimOp(IndexKind indexKind, const std::string& debug)
        : Param(indexKind, debug)
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

    ArithOpKind arithOpKind() { return (ArithOpKind) indexKind(); }
    const Use& luse() const { return luse_; }
    const Use& ruse() const { return ruse_; }

    virtual uint64_t hash() const;

private:

    Use luse_;
    Use ruse_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_PRIMOP_H
