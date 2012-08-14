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

    PrimOp(int kind, const Type* type, size_t numOps)
        : Def(kind, type, numOps)
    {}
};

//------------------------------------------------------------------------------

class BinOp : public PrimOp {
protected:

    BinOp(NodeKind kind, const Type* type, const Def* lhs, const Def* rhs)
        : PrimOp(kind, type, 2)
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
        : BinOp((NodeKind) kind, lhs->type(), lhs, rhs)
    {}

public:

    ArithOpKind arithop_kind() const { return (ArithOpKind) node_kind(); }

    static bool isDiv(ArithOpKind kind) { 
        return  kind == ArithOp_sdiv 
             || kind == ArithOp_udiv
             || kind == ArithOp_fdiv; 
    }
    static bool isRem(ArithOpKind kind) { 
        return  kind == ArithOp_srem 
             || kind == ArithOp_urem
             || kind == ArithOp_frem; 
    }
    static bool isBit(ArithOpKind kind) {
        return  kind == ArithOp_and
             || kind == ArithOp_or
             || kind == ArithOp_xor;
    }
    static bool isShift(ArithOpKind kind) {
        return  kind == ArithOp_shl
             || kind == ArithOp_lshr
             || kind == ArithOp_ashr;
    }
    static bool isDivOrRem(ArithOpKind kind) { return isDiv(kind) || isRem(kind); }

    bool isDiv()      const { return isDiv  (arithop_kind()); }
    bool isRem()      const { return isRem  (arithop_kind()); }
    bool isBit()      const { return isBit  (arithop_kind()); }
    bool isShift()    const { return isShift(arithop_kind()); }
    bool isDivOrRem() const { return isDivOrRem(arithop_kind()); }
    bool isCommutative() const { return isCommutative(arithop_kind()); }

    static bool isCommutative(ArithOpKind kind) {
        return kind == ArithOp_add
            || kind == ArithOp_mul
            || kind == ArithOp_fadd
            || kind == ArithOp_fmul
            || kind == ArithOp_and
            || kind == ArithOp_or
            || kind == ArithOp_xor;
    }

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(RelOpKind kind, const Def* lhs, const Def* rhs);

public:

    RelOpKind relop_kind() const { return (RelOpKind) node_kind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public PrimOp {
private:

    ConvOp(ConvOpKind kind, const Def* from, const Type* to)
        : PrimOp(kind, to, 1)
        , from_(from)
        , to_(to)
    {
        setOp(0, from);
    }

public:

    const Def* from() const { return op(0); }
    const Type* to() const { return to_; }
    ConvOpKind convop_kind() const { return (ConvOpKind) node_kind(); }

private:

    const Def* from_;
    const Type* to_;

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer &printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Select : public PrimOp {
private:

    Select(const Def* cond, const Def* t, const Def* f);

public:

    const Def* cond() const { return op(0); }
    const Def* tval() const { return op(1); }
    const Def* fval() const { return op(2); }

    virtual void vdump(Printer &printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class TupleOp : public PrimOp {
protected:

    TupleOp(NodeKind kind, const Type* type, size_t numOps, const Def* tuple, u32 index);

public:

    const Def* tuple() const { return op(0); }
    u32 index() const { return index_; }

private:

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

class Extract : public TupleOp {
private:

    Extract(const Def* tuple, u32 index);
    
public:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Insert : public TupleOp {
private:

    Insert(const Def* tuple, u32 index, const Def* value);
    
public:

    const Def* value() const { return op(1); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public PrimOp {
private:

    Tuple(World& world, ArrayRef<const Def*> args);

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
