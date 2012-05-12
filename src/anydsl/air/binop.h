#ifndef ANYDSL_BINOP_H
#define ANYDSL_BINOP_H

#include "anydsl/air/enums.h"
#include "anydsl/air/def.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

class BinOp : public PrimOp {
protected:

    BinOp(IndexKind index, Type* type,
            Def* ldef, Def* rdef, 
            const std::string& ldebug, const std::string& rdebug,
            const std::string& debug)
        : PrimOp(index, type, debug)
        , luse_(ldef, this, ldebug)
        , ruse_(rdef, this, rdebug)
    {}

public:

    const Use& luse() const { return luse_; }
    const Use& ruse() const { return ruse_; }

    ArithOpKind arithOpKind() { return (ArithOpKind) index(); }
    virtual uint64_t hash() const;

protected:

    Use luse_;
    Use ruse_;
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:

    ArithOp(ArithOpKind arithOpKind, 
            Def* ldef, Def* rdef, 
            const std::string& ldebug, const std::string& rdebug,
            const std::string& debug)
        : BinOp((IndexKind) arithOpKind, ldef->type(), ldef, rdef, ldebug, rdebug, debug)
    {
        anydsl_assert(ldef->type() == rdef->type(), "type are not equal");
    }

    friend class Universe;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(ArithOpKind arithOpKind, 
            Def* ldef, Def* rdef, 
            const std::string& ldebug, const std::string& rdebug,
            const std::string& debug);

    friend class Universe;
};

//------------------------------------------------------------------------------


} // namespace anydsl

#endif // ANYDSL_PRIMOP_H
