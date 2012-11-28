#ifndef ANYDSL2_PRIMOP_H
#define ANYDSL2_PRIMOP_H

#include <boost/functional/hash.hpp>
#include <boost/tuple/tuple.hpp>

#include "anydsl2/enums.h"
#include "anydsl2/def.h"

#define ANYDSL2_HASH_EQUAL \
    virtual bool equal(const PrimOp* other) const { return equal_op(tuple(), other); } \
    virtual size_t hash() const { return hash_op(tuple()); }

namespace anydsl2 {

class PrimLit;

//------------------------------------------------------------------------------

typedef boost::tuple<int, const Type*> DefTuple0;
typedef boost::tuple<int, const Type*, const Def*> DefTuple1;
typedef boost::tuple<int, const Type*, const Def*, const Def*> DefTuple2;
typedef boost::tuple<int, const Type*, const Def*, const Def*, const Def*> DefTuple3;
typedef boost::tuple<int, const Type*, ArrayRef<const Def*> > DefTupleN;

template<class T>
inline size_t hash_kind_type_size(const T& tuple, size_t size) {
    size_t seed = 0;
    boost::hash_combine(seed, size);
    boost::hash_combine(seed, tuple.template get<0>());
    boost::hash_combine(seed, tuple.template get<1>());
    return seed;
}
size_t hash_op(const DefTuple0&);
size_t hash_op(const DefTuple1&);
size_t hash_op(const DefTuple2&);
size_t hash_op(const DefTuple3&);
size_t hash_op(const DefTupleN&);

bool equal_op(const DefTuple0&, const PrimOp*);
bool equal_op(const DefTuple1&, const PrimOp*);
bool equal_op(const DefTuple2&, const PrimOp*);
bool equal_op(const DefTuple3&, const PrimOp*);
bool equal_op(const DefTupleN&, const PrimOp*);

//------------------------------------------------------------------------------

class PrimOp : public Def {
protected:

    PrimOp(int kind, size_t size, const Type* type, const std::string& name)
        : Def(kind, size, type, name)
    {}

    DefTupleN tuple() const { return DefTupleN(kind(), type(), ops()); }
    ANYDSL2_HASH_EQUAL

    friend class PrimOpHash;
    friend class PrimOpEqual;
};

struct PrimOpHash : std::unary_function<const PrimOp*, size_t> {
    size_t operator () (const PrimOp* o) const { return o->hash(); }
};

struct PrimOpEqual : std::binary_function<const PrimOp*, const PrimOp*, bool> {
    bool operator () (const PrimOp* o1, const PrimOp* o2) const { return o1->equal(o2); }
};

//------------------------------------------------------------------------------

class BinOp : public PrimOp {
protected:

    BinOp(NodeKind kind, const Type* type, const Def* lhs, const Def* rhs, const std::string& name)
        : PrimOp(kind, 2, type, name)
    {
        assert(lhs->type() == rhs->type() && "types are not equal");
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

    ArithOp(ArithOpKind kind, const Def* lhs, const Def* rhs, const std::string& name)
        : BinOp((NodeKind) kind, lhs->type(), lhs, rhs, name)
    {}

public:

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

    RelOp(RelOpKind kind, const Def* lhs, const Def* rhs, const std::string& name);

public:

    RelOpKind relop_kind() const { return (RelOpKind) node_kind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public PrimOp {
private:

    ConvOp(ConvOpKind kind, const Type* to, const Def* from, const std::string& name)
        : PrimOp(kind, 1, to, name)
    {
        set_op(0, from);
    }

public:

    const Def* from() const { return op(0); }
    ConvOpKind convop_kind() const { return (ConvOpKind) node_kind(); }

private:

    virtual void vdump(Printer &printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Select : public PrimOp {
private:

    Select(const Def* cond, const Def* t, const Def* f, const std::string& name);

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

    TupleOp(NodeKind kind, size_t size, const Type* type, const Def* tuple, const Def* index, const std::string& name);

public:

    const Def* tuple() const { return op(0); }
    const Def* index() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

class Extract : public TupleOp {
private:

    Extract(const Def* tuple, const Def* index, const std::string& name);
    
    virtual void vdump(Printer& printer) const;

public:

    friend class World;
};

//------------------------------------------------------------------------------

class Insert : public TupleOp {
private:

    Insert(const Def* tuple, const Def* index, const Def* value, const std::string& name);
    
public:

    const Def* value() const { return op(2); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public PrimOp {
private:

    Tuple(World& world, ArrayRef<const Def*> args, const std::string& name);

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
