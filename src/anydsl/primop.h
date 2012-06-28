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

    BinOp(IndexKind index, const Type* type, const Def* ldef, const Def* rdef)
        : PrimOp(index, type, 2)
    {
        anydsl_assert(ldef->type() == rdef->type(), "types are not equal");
        setOp(0, ldef);
        setOp(1, rdef);
    }

    virtual void dump(Printer &printer, bool descent) const;

public:

    const Def* ldef() const { return ops_[0]; }
    const Def* rdef() const { return ops_[1]; }
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:

    ArithOp(ArithOpKind kind, const Def* ldef, const Def* rdef)
        : BinOp((IndexKind) kind, ldef->type(), ldef, rdef)
    {}

public:

    ArithOpKind kind() { return (ArithOpKind) indexKind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(RelOpKind kind, const Def* ldef, const Def* rdef);

public:

    RelOpKind kind() { return (RelOpKind) indexKind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class Select : public PrimOp {
private:

    Select(const Def* cond, const Def* t, const Def* f);

public:

    const Def* cond() const { return ops_[0]; }
    const Def* tdef() const { return ops_[1]; }
    const Def* fdef() const { return ops_[2]; }

    RelOpKind kind() { return (RelOpKind) indexKind(); }

    virtual void dump(Printer &printer, bool descent) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Extract : public PrimOp {
private:

    Extract(const Def* tuple, const PrimLit* elem);
    
public:

    const Def* tuple() const { return ops_[0]; }
    const Def* elem()  const { return ops_[1]; }

    virtual void dump(Printer& printer, bool descent) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Insert : public PrimOp {
private:

    Insert(const Def* tuple, const PrimLit* elem, const Def* value);
    
public:

    const Def* tuple() const { return ops_[0]; }
    const Def* elem()  const { return ops_[1]; }
    const Def* value() const { return ops_[2]; }

    virtual void dump(Printer& printer, bool descent) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public PrimOp {
private:

    Tuple(World& world, const Def* const* begin, const Def* const* end);

public:

    virtual void dump(Printer& printer, bool descent) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
