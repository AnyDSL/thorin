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
    PrimOp(size_t size, int kind, const Type* type, const std::string& name)
        : DefNode(-1, kind, size, type, true, name)
    {}

public:
    void update(size_t i, const DefNode* with);
    virtual const char* op_name() const;
    virtual size_t hash() const { return hash_combine(DefNode::hash(), type()); }
    virtual bool equal(const Node* other) const { 
        return DefNode::equal(other) ? type() == other->as<PrimOp>()->type() : false; 
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
    VectorOp(size_t size, NodeKind kind, const Type* type, const DefNode* cond, const std::string& name);

public:
    const DefNode* cond() const { return op(0); }
};

//------------------------------------------------------------------------------

class Select : public VectorOp {
private:
    Select(const DefNode* cond, const DefNode* tval, const DefNode* fval, const std::string& name);

public:
    const DefNode* tval() const { return op(1); }
    const DefNode* fval() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------


class BinOp : public VectorOp {
protected:
    BinOp(NodeKind kind, const Type* type, const DefNode* cond, const DefNode* lhs, const DefNode* rhs, const std::string& name);

public:
    const DefNode* lhs() const { return op(1); }
    const DefNode* rhs() const { return op(2); }
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:
    ArithOp(ArithOpKind kind, const DefNode* cond, const DefNode* lhs, const DefNode* rhs, const std::string& name)
        : BinOp((NodeKind) kind, lhs->type(), cond, lhs, rhs, name)
    {}

public:
    ArithOpKind arithop_kind() const { return (ArithOpKind) node_kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:
    RelOp(RelOpKind kind, const DefNode* cond, const DefNode* lhs, const DefNode* rhs, const std::string& name);

public:
    RelOpKind relop_kind() const { return (RelOpKind) node_kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public VectorOp {
private:
    ConvOp(ConvOpKind kind, const DefNode* cond, const DefNode* from, const Type* to, const std::string& name)
        : VectorOp(2, (NodeKind) kind, to, cond, name)
    {
        set_op(1, from);
    }

public:
    const DefNode* from() const { return op(1); }
    ConvOpKind convop_kind() const { return (ConvOpKind) node_kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class TupleOp : public PrimOp {
protected:
    TupleOp(size_t size, int kind, const Type* type, const DefNode* tuple, const DefNode* index, const std::string& name)
        : PrimOp(size, kind, type, name)
    {
        set_op(0, tuple);
        set_op(1, index);
    }

public:
    const DefNode* tuple() const { return op(0); }
    const DefNode* index() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

class TupleExtract : public TupleOp {
private:
    TupleExtract(const DefNode* tuple, const DefNode* index, const std::string& name);
    
    friend class World;
};

//------------------------------------------------------------------------------

class TupleInsert : public TupleOp {
private:
    TupleInsert(const DefNode* tuple, const DefNode* index, const DefNode* value, const std::string& name);

public:
    const DefNode* value() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public PrimOp {
private:
    Tuple(World& world, ArrayRef<const DefNode*> args, const std::string& name);

    friend class World;
};

//------------------------------------------------------------------------------

class Vector : public PrimOp {
private:
    Vector(World& world, ArrayRef<const DefNode*> args, const std::string& name);

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
