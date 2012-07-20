#ifndef ANYDSL_PRIMOP_H
#define ANYDSL_PRIMOP_H

#include <boost/array.hpp>

#include "anydsl/enums.h"
#include "anydsl/def.h"

namespace anydsl {

class PrimLit;

//------------------------------------------------------------------------------

class PrimOp : public Def {
protected:

    PrimOp(IndexKind index, const Type* type, size_t numOps)
        : Def(index, type, numOps)
    {}

public:

    PrimOpKind primOpKind() const { return (PrimOpKind) indexKind(); }
};

//------------------------------------------------------------------------------

class BinOp : public PrimOp {
protected:

    BinOp(IndexKind index, const Type* type, const Def* lhs, const Def* rhs)
        : PrimOp(index, type, 2)
    {
        anydsl_assert(lhs->type() == rhs->type(), "types are not equal");
        setOp(0, lhs);
        setOp(1, rhs);
    }

public:

    const Def* lhs() const { return op(0); }
    const Def* rhs() const { return op(1); }

private:

    virtual void vdump(Printer &printer) const;
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:

    ArithOp(ArithOpKind kind, const Def* lhs, const Def* rhs)
        : BinOp((IndexKind) kind, lhs->type(), lhs, rhs)
    {}

public:

    ArithOpKind arithOpKind() const { return (ArithOpKind) indexKind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(RelOpKind kind, const Def* lhs, const Def* rhs);

public:

    RelOpKind relOpKind() const { return (RelOpKind) indexKind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class Select : public PrimOp {
private:

    Select(const Def* cond, const Def* t, const Def* f);

public:

    const Def* cond() const { return op(0); }
    const Def* tdef() const { return op(1); }
    const Def* fdef() const { return op(2); }

    RelOpKind kind() { return (RelOpKind) indexKind(); }

    virtual void vdump(Printer &printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class TupleOp : public PrimOp {
protected:

    TupleOp(IndexKind indexKind, const Type* type, size_t numOps, const Def* tuple, size_t index);

public:

    const Def* tuple() const { return op(0); }
    size_t index() const { return index_; }

private:

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

class Extract : public TupleOp {
private:

    Extract(const Def* tuple, size_t index);
    
public:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Insert : public TupleOp {
private:

    Insert(const Def* tuple, size_t index, const Def* value);
    
public:

    const Def* value() const { return op(1); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public PrimOp {
private:

    Tuple(World& world, const Def* const* begin, const Def* const* end);

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
