#ifndef ANYDSL2_PRIMOP_H
#define ANYDSL2_PRIMOP_H

#include <boost/tuple/tuple_comparison.hpp>

#include "anydsl2/enums.h"
#include "anydsl2/def.h"
#include "anydsl2/util/hash.h"

#define ANYDSL2_HASH_EQUAL \
    virtual bool equal(const PrimOp* other) const { \
        typedef BOOST_TYPEOF(*this) T; \
        return other->isa<T>() && this->as_tuple() == other->as<T>()->as_tuple(); \
    } \
    virtual size_t hash() const { return hash_tuple(as_tuple()); }

namespace anydsl2 {

class PrimLit;

//------------------------------------------------------------------------------

typedef boost::tuple<int, const Type*> DefTuple0;
typedef boost::tuple<int, const Type*, const Def*> DefTuple1;
typedef boost::tuple<int, const Type*, const Def*, const Def*> DefTuple2;
typedef boost::tuple<int, const Type*, const Def*, const Def*, const Def*> DefTuple3;
typedef boost::tuple<int, const Type*, ArrayRef<const Def*> > DefTupleN;


//------------------------------------------------------------------------------

class PrimOp : public Def {
protected:

    PrimOp(size_t size, int kind, const Type* type, const std::string& name)
        : Def(kind, size, type, name)
    {}

public:

    DefTupleN as_tuple() const { return DefTupleN(kind(), type(), ops()); }
    ANYDSL2_HASH_EQUAL

    friend class PrimOpHash;
    friend class PrimOpEqual;
};

template<class T, class D> inline
bool smart_eq(const T& t, const PrimOp* primop) { return smart_eq(t, primop->as<D>()->as_tuple()); }

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
        : PrimOp(2, kind, type, name)
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

    ArithOp(const DefTuple2& tuple, const std::string& name)
        : BinOp((NodeKind) tuple.get<0>(), tuple.get<1>(), tuple.get<2>(), tuple.get<3>(), name)
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

    RelOp(const DefTuple2& tuple, const std::string& name)
        : BinOp((NodeKind) tuple.get<0>(), tuple.get<1>(), tuple.get<2>(), tuple.get<3>(), name)
    {}

public:

    RelOpKind relop_kind() const { return (RelOpKind) node_kind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public PrimOp {
private:

    ConvOp(const DefTuple1& tuple, const std::string& name)
        : PrimOp(1, (NodeKind) tuple.get<0>(), tuple.get<1>(), name)
    {
        set_op(0, tuple.get<2>());
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

    Select(const DefTuple3& tuple, const std::string& name);

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

    TupleOp(size_t size, int kind, const Type* type, const Def* agg, const Def* index, const std::string& name)
        : PrimOp(size, kind, type, name)
    {
        set_op(0, agg);
        set_op(1, index);
    }

public:

    const Def* tuple() const { return op(0); }
    const Def* index() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

class Extract : public TupleOp {
private:

    Extract(const DefTuple2& tuple, const std::string& name)
        : TupleOp(2, tuple.get<0>(), tuple.get<1>(), tuple.get<2>(), tuple.get<3>(), name)
    {}
    
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
