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

    PrimOp(int kind, size_t size, const Type* type)
        : Def(kind, size, type)
    {}

public:

    virtual PrimOp* clone() const = 0;
};

//------------------------------------------------------------------------------

class BinOp : public PrimOp {
protected:

    BinOp(NodeKind kind, const Type* type, const Def* lhs, const Def* rhs)
        : PrimOp(kind, 2, type)
    {
        anydsl_assert(lhs->type() == rhs->type(), "types are not equal");
        set_op(0, lhs);
        set_op(1, rhs);
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

    virtual ArithOp* clone() const { return new ArithOp(*this); }

    ArithOpKind arithop_kind() const { return (ArithOpKind) node_kind(); }

    static bool is_div(ArithOpKind kind) { 
        return  kind == ArithOp_sdiv 
             || kind == ArithOp_udiv
             || kind == ArithOp_fdiv; 
    }
    static bool is_rem(ArithOpKind kind) { 
        return  kind == ArithOp_srem 
             || kind == ArithOp_urem
             || kind == ArithOp_frem; 
    }
    static bool is_bit(ArithOpKind kind) {
        return  kind == ArithOp_and
             || kind == ArithOp_or
             || kind == ArithOp_xor;
    }
    static bool is_shift(ArithOpKind kind) {
        return  kind == ArithOp_shl
             || kind == ArithOp_lshr
             || kind == ArithOp_ashr;
    }
    static bool is_div_or_rem(ArithOpKind kind) { return is_div(kind) || is_rem(kind); }
    static bool is_commutative(ArithOpKind kind) {
        return kind == ArithOp_add
            || kind == ArithOp_mul
            || kind == ArithOp_fadd
            || kind == ArithOp_fmul
            || kind == ArithOp_and
            || kind == ArithOp_or
            || kind == ArithOp_xor;
    }

    bool is_div()      const { return is_div  (arithop_kind()); }
    bool is_rem()      const { return is_rem  (arithop_kind()); }
    bool is_bit()      const { return is_bit  (arithop_kind()); }
    bool is_shift()    const { return is_shift(arithop_kind()); }
    bool is_div_or_rem() const { return is_div_or_rem(arithop_kind()); }
    bool is_commutative() const { return is_commutative(arithop_kind()); }

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(RelOpKind kind, const Def* lhs, const Def* rhs);
    virtual RelOp* clone() const { return new RelOp(*this); }

public:

    RelOpKind relop_kind() const { return (RelOpKind) node_kind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public PrimOp {
private:

    ConvOp(ConvOpKind kind, const Type* to, const Def* from)
        : PrimOp(kind, 1, to)
    {
        set_op(0, from);
    }
    virtual ConvOp* clone() const { return new ConvOp(*this); }

public:

    const Def* from() const { return op(0); }
    ConvOpKind convop_kind() const { return (ConvOpKind) node_kind(); }

private:

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;
    virtual void vdump(Printer &printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Select : public PrimOp {
private:

    Select(const Def* cond, const Def* t, const Def* f);

public:

    virtual Select* clone() const { return new Select(*this); }

    const Def* cond() const { return op(0); }
    const Def* tval() const { return op(1); }
    const Def* fval() const { return op(2); }

    virtual void vdump(Printer &printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class TupleOp : public PrimOp {
protected:

    TupleOp(NodeKind kind, size_t size, const Type* type, const Def* tuple, u32 index);
    TupleOp(const TupleOp& tuple)
        : PrimOp(tuple)
        , index_(tuple.index())
    {}

public:

    const Def* tuple() const { return op(0); }
    u32 index() const { return index_; }

private:

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;

    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

class Extract : public TupleOp {
private:

    Extract(const Def* tuple, u32 index);
    
    virtual void vdump(Printer& printer) const;

public:

    virtual Extract* clone() const { return new Extract(*this); }

    friend class World;
};

//------------------------------------------------------------------------------

class Insert : public TupleOp {
private:

    Insert(const Def* tuple, u32 index, const Def* value);
    virtual Insert* clone() const { return new Insert(*this); }
    
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
    virtual Tuple* clone() const { return new Tuple(*this); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
