#ifndef ANYDSL_PRIMOP_H
#define ANYDSL_PRIMOP_H

#include "anydsl/air/enums.h"
#include "anydsl/air/def.h"
#include "anydsl/air/use.h"

namespace anydsl {

class PrimConst;

//------------------------------------------------------------------------------

class PrimOp : public Value {
public:

    PrimOpKind primOpKind() const { return (PrimOpKind) index(); }

    bool compare(PrimOp* other) const;

protected:

    PrimOp(IndexKind index, Type* type, const std::string& debug)
        : Value(index, type, debug)
    {}
};

//------------------------------------------------------------------------------

class ArithOp : public PrimOp {
private:

    ArithOp(ArithOpKind arithOpKind, 
            Def* ldef, Def* rdef, 
            const std::string& ldebug, const std::string& rdebug,
            const std::string& debug)
        : PrimOp((IndexKind) arithOpKind, ldef->type(), debug)
        , luse_(ldef, this, ldebug)
        , ruse_(rdef, this, rdebug)
    {
        anydsl_assert(ldef->type() == rdef->type(), "type are not equal");
    }

public:

    ArithOpKind arithOpKind() { return (ArithOpKind) index(); }
    const Use& luse() const { return luse_; }
    const Use& ruse() const { return ruse_; }

    virtual uint64_t hash() const;

private:

    Use luse_;
    Use ruse_;

    friend class Universe;
};

//------------------------------------------------------------------------------

class Extract : public PrimOp {
private:

    Extract(Def* tuple, PrimConst* elem, 
            const std::string& tupleDebug,
            const std::string debug);

public:

    const Use& tuple() const { return tuple_; }
    const PrimConst* elem() const { return elem_; }

    virtual uint64_t hash() const;

private:

    Use tuple_;
    PrimConst* elem_;
};

//------------------------------------------------------------------------------

class Insert : public PrimOp {
private:

    Insert(Def* tuple, Def* value, PrimConst* elem,
           const std::string& tupleDebug, const std::string& valueDebug,
           const std::string debug);

public:

    const Use& tuple() const { return tuple_; }
    const Use& value() const { return value_; }
    const PrimConst* elem() const { return elem_; }

private:

    Use tuple_;
    Use value_;
    PrimConst* elem_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_PRIMOP_H
