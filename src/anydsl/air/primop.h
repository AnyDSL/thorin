#ifndef ANYDSL_PRIMOP_H
#define ANYDSL_PRIMOP_H

#include <boost/array.hpp>

#include "anydsl/air/enums.h"
#include "anydsl/air/def.h"
#include "anydsl/air/use.h"

namespace anydsl {

class PrimLit;

//------------------------------------------------------------------------------

class PrimOp : public Value {
public:

    PrimOpKind primOpKind() const { return (PrimOpKind) index(); }

    const Ops& ops() { return ops_; }

protected:

    PrimOp(IndexKind index, const Type* type)
        : Value(index, type)
    {}

    Ops ops_;
};

//------------------------------------------------------------------------------

class BinOp : public PrimOp {
protected:

    BinOp(IndexKind index, const Type* type, Def* ldef, Def* rdef)
        : PrimOp(index, type)
        , luse(this, ldef)
        , ruse(this, rdef)
    {
        anydsl_assert(ldef->type() == rdef->type(), "types are not equal");
    }

    static ValueNumber VN(IndexKind kind, Def* ldef, Def* rdef) {
        return ValueNumber(kind, (uintptr_t) ldef, (uintptr_t) rdef);
    }

public:

    typedef boost::array<Use*, 2> LRUse;
    typedef boost::array<const Use*, 2> ConstLRUse;

    LRUse lruse() { return (LRUse){{ &luse, &ruse }}; }
    ConstLRUse lruse() const { return (ConstLRUse){{ &ruse, &ruse }}; }

public:

    Use luse;
    Use ruse;
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:

    ArithOp(const ValueNumber& vn)
        : BinOp((IndexKind) vn.index, 
                ((Def*) vn.op1)->type(), 
                (Def*) vn.op1, 
                (Def*) vn.op2)
    {}

public:

    ArithOpKind kind() { return (ArithOpKind) index(); }

    static ValueNumber VN(ArithOpKind kind, Def* ldef, Def* rdef) {
        return BinOp::VN((IndexKind) kind, ldef, rdef);
    }

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(const ValueNumber& vn);

public:

    RelOpKind kind() { return (RelOpKind) index(); }

    static ValueNumber VN(RelOpKind kind, Def* ldef, Def* rdef) {
        return ValueNumber((IndexKind) kind, (uintptr_t) ldef, (uintptr_t) rdef);
    }

    friend class World;
};

//------------------------------------------------------------------------------

class SigmaOp : public PrimOp {
protected:

    SigmaOp(IndexKind index, const Type* type, Def* tuple, PrimLit* elem);

public:

    Use tuple;

private:

    PrimLit* elem_;
};

//------------------------------------------------------------------------------

class Extract : public SigmaOp {
private:

    Extract(Def* tuple, PrimLit* elem);
};

//------------------------------------------------------------------------------

class Insert : public SigmaOp {
private:

    Insert(Def* tuple, PrimLit* elem, Def* value)
        : SigmaOp(Index_Insert, tuple->type(), tuple, elem)
        , value(value, this)
    {}

public:

    Use value;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_PRIMOP_H
