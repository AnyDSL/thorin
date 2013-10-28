#ifndef ANYDSL2_PRIMOP_H
#define ANYDSL2_PRIMOP_H

#include "anydsl2/enums.h"
#include "anydsl2/def.h"
#include "anydsl2/util/hash.h"

namespace anydsl2 {

class PrimLit;

//------------------------------------------------------------------------------

class PrimOp : public DefNode {
protected:
    PrimOp(size_t size, NodeKind kind, const Type* type, const std::string& name)
        : DefNode(-1, kind, size, type, true, name)
    {}

public:
    void update(size_t i, Def with);
    virtual const char* op_name() const;
    virtual size_t hash() const {
        size_t seed = hash_combine(hash_combine(hash_value((int) kind()), size()), type());
        for (auto op : ops_)
            seed = hash_combine(seed, op.node());
        return seed;
    }
    virtual bool equal(const PrimOp* other) const {
        bool result = this->kind() == other->kind() && this->size() == other->size() && this->type() == other->type();
        for (size_t i = 0, e = size(); result && i != e; ++i)
            result &= this->ops_[i] == other->ops_[i];
        return result;
    }

private:
    void set_gid(size_t gid) const { const_cast<size_t&>(const_cast<PrimOp*>(this)->gid_) = gid; }

    friend struct PrimOpHash;
    friend struct PrimOpEqual;
    friend class World;
};

struct PrimOpHash { size_t operator () (const PrimOp* o) const { return o->hash(); } };
struct PrimOpEqual { bool operator () (const PrimOp* o1, const PrimOp* o2) const { return o1->equal(o2); } };

//------------------------------------------------------------------------------

class VectorOp : public PrimOp {
protected:
    VectorOp(size_t size, NodeKind kind, const Type* type, Def cond, const std::string& name);

public:
    Def cond() const { return op(0); }
};

//------------------------------------------------------------------------------

class Select : public VectorOp {
private:
    Select(Def cond, Def tval, Def fval, const std::string& name);

public:
    Def tval() const { return op(1); }
    Def fval() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------


class BinOp : public VectorOp {
protected:
    BinOp(NodeKind kind, const Type* type, Def cond, Def lhs, Def rhs, const std::string& name);

public:
    Def lhs() const { return op(1); }
    Def rhs() const { return op(2); }
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:
    ArithOp(ArithOpKind kind, Def cond, Def lhs, Def rhs, const std::string& name)
        : BinOp((NodeKind) kind, lhs->type(), cond, lhs, rhs, name)
    {}

public:
    ArithOpKind arithop_kind() const { return (ArithOpKind) kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:
    RelOp(RelOpKind kind, Def cond, Def lhs, Def rhs, const std::string& name);

public:
    RelOpKind relop_kind() const { return (RelOpKind) kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public VectorOp {
private:
    ConvOp(ConvOpKind kind, Def cond, Def from, const Type* to, const std::string& name)
        : VectorOp(2, (NodeKind) kind, to, cond, name)
    {
        set_op(1, from);
    }

public:
    Def from() const { return op(1); }
    ConvOpKind convop_kind() const { return (ConvOpKind) kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class TupleOp : public PrimOp {
protected:
    TupleOp(size_t size, NodeKind kind, const Type* type, Def tuple, Def index, const std::string& name)
        : PrimOp(size, kind, type, name)
    {
        set_op(0, tuple);
        set_op(1, index);
    }

public:
    Def tuple() const { return op(0); }
    Def index() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

class TupleExtract : public TupleOp {
private:
    TupleExtract(Def tuple, Def index, const std::string& name);
    
    friend class World;
};

//------------------------------------------------------------------------------

class TupleInsert : public TupleOp {
private:
    TupleInsert(Def tuple, Def index, Def value, const std::string& name);

public:
    Def value() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public PrimOp {
private:
    Tuple(World& world, ArrayRef<Def> args, const std::string& name);

    friend class World;
};

//------------------------------------------------------------------------------

class Vector : public PrimOp {
private:
    Vector(World& world, ArrayRef<Def> args, const std::string& name);

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
