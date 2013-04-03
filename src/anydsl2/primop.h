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
        : Def(-1, kind, size, type, true, name)
    {}

public:

    DefTupleN as_tuple() const { return DefTupleN(kind(), type(), ops()); }
    void update(size_t i, const Def* with);
    virtual void vdump(Printer &printer) const;
    virtual const char* op_name() const;
    ANYDSL2_HASH_EQUAL

private:

    void set_gid(size_t gid) const { const_cast<size_t&>(const_cast<PrimOp*>(this)->gid_) = gid; }

    friend struct PrimOpHash;
    friend struct PrimOpEqual;
    friend class World;
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
        : PrimOp(2, kind, type, name)
    {
        assert(lhs->type() == rhs->type() && "types are not equal");
        set_op(0, lhs);
        set_op(1, rhs);
    }

public:

    const Def* lhs() const { return op(0); }
    const Def* rhs() const { return op(1); }
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:

    ArithOp(const DefTuple2& args, const std::string& name)
        : BinOp((NodeKind) args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
    {}

public:

    ArithOpKind arithop_kind() const { return (ArithOpKind) node_kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(const DefTuple2& args, const std::string& name)
        : BinOp((NodeKind) args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
    {}

public:

    RelOpKind relop_kind() const { return (RelOpKind) node_kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public PrimOp {
private:

    ConvOp(const DefTuple1& args, const std::string& name)
        : PrimOp(1, (NodeKind) args.get<0>(), args.get<1>(), name)
    {
        set_op(0, args.get<2>());
    }

public:

    const Def* from() const { return op(0); }
    ConvOpKind convop_kind() const { return (ConvOpKind) node_kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Select : public PrimOp {
private:

    Select(const DefTuple3& args, const std::string& name);

public:

    const Def* cond() const { return op(0); }
    const Def* tval() const { return op(1); }
    const Def* fval() const { return op(2); }

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

    Extract(const DefTuple2& args, const std::string& name)
        : TupleOp(2, args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
    {}
    
    friend class World;
};

//------------------------------------------------------------------------------

class Insert : public TupleOp {
private:

    Insert(const DefTuple3& args, const std::string& name)
        : TupleOp(3, args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
    {
        set_op(2, args.get<4>());
    }
    
public:

    const Def* value() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public PrimOp {
private:

    Tuple(const DefTupleN& args, const std::string& name);

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
