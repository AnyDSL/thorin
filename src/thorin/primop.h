#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/enums.h"
#include "thorin/def.h"
#include "thorin/util/hash.h"

namespace thorin {

//------------------------------------------------------------------------------

class PrimOp : public DefNode {
protected:
    PrimOp(size_t size, NodeKind kind, Type type, const std::string& name)
        : DefNode(-1, kind, size, type ? type.unify() : nullptr, true, name)
    {}

    void set_type(Type type) { type_ = type.unify(); }

public:
    virtual const char* op_name() const;
    virtual size_t hash() const;
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
    VectorOp(size_t size, NodeKind kind, Type type, Def cond, const std::string& name);

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
    BinOp(NodeKind kind, Type type, Def cond, Def lhs, Def rhs, const std::string& name);

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

class Cmp : public BinOp {
private:
    Cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const std::string& name);

public:
    CmpKind cmp_kind() const { return (CmpKind) kind(); }
    virtual const char* op_name() const;

    friend class World;
};

//------------------------------------------------------------------------------

class ConvOp : public VectorOp {
protected:
    ConvOp(NodeKind kind, Def cond, Def from, Type to, const std::string& name)
        : VectorOp(2, kind, to, cond, name)
    {
        set_op(1, from);
    }

public:
    Def from() const { return op(1); }
};

class Cast : public ConvOp {
private:
    Cast(Def cond, Def from, Type to, const std::string& name)
        : ConvOp(Node_Cast, cond, from, to, name)
    {}

    friend class World;
};

class Bitcast : public ConvOp {
private:
    Bitcast(Def cond, Def from, Type to, const std::string& name)
        : ConvOp(Node_Bitcast, cond, from, to, name)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

class Aggregate : public PrimOp {
protected:
    Aggregate(NodeKind kind, ArrayRef<Def> args, const std::string& name)
        : PrimOp(args.size(), kind, Type() /*set later*/, name)
    {
        for (size_t i = 0, e = size(); i != e; ++i)
            set_op(i, args[i]);
    }
};

class DefiniteArray : public Aggregate {
private:
    DefiniteArray(World& world, Type elem, ArrayRef<Def> args, const std::string& name);

public:
    DefiniteArrayType type() const { return Aggregate::type().as<DefiniteArrayType>(); }
    Type elem_type() const;

    friend class World;
};

class Tuple : public Aggregate {
private:
    Tuple(World& world, ArrayRef<Def> args, const std::string& name);

public:
    TupleType tuple_type() const { return Aggregate::type().as<TupleType>(); }

    friend class World;
};

class Vector : public Aggregate {
private:
    Vector(World& world, ArrayRef<Def> args, const std::string& name);
    friend class World;
};

//------------------------------------------------------------------------------

class AggOp : public PrimOp {
protected:
    AggOp(size_t size, NodeKind kind, Type type, Def agg, Def index, const std::string& name)
        : PrimOp(size, kind, type, name)
    {
        set_op(0, agg);
        set_op(1, index);
    }

public:
    Def agg() const { return op(0); }
    Def index() const { return op(1); }

    friend class World;
};

class Extract : public AggOp {
private:
    Extract(Def agg, Def index, const std::string& name);

public:
    static Type type(Def agg, Def index);

    friend class World;
};

class Insert : public AggOp {
private:
    Insert(Def agg, Def index, Def value, const std::string& name);

public:
    Def value() const { return op(2); }
    static Type type(Def agg);

    friend class World;
};

//------------------------------------------------------------------------------

/**
 * Loads Effective Address.
 * Takes a pointer \p ptr to an aggregate as input.
 * Then, the address to the \p index'th element is computed.
 */
class LEA : public PrimOp {
private:
    LEA(Def ptr, Def index, const std::string& name);

public:
    Def ptr() const { return op(0); }
    Def index() const { return op(1); }

    PtrType ptr_type() const; ///< Returns the ptr type from \p ptr().
    Type referenced_type() const; ///< Returns the type referenced by \p ptr().

    friend class World;
};

//------------------------------------------------------------------------------

class EvalOp : public PrimOp {
protected:
    EvalOp(NodeKind kind, Def def, const std::string& name);

public:
    Def def() const { return op(0); }
};

class Run : public EvalOp {
private:
    Run(Def def, const std::string& name)
        : EvalOp(Node_Run, def, name)
    {}

    friend class World;
};

class Hlt : public EvalOp {
private:
    Hlt(Def def, const std::string& name)
        : EvalOp(Node_Hlt, def, name)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

/**
 * This represents a slot in a stack frame opend via \p Enter.
 * Loads from this address yield \p Bottom if the frame has already been closed via \p Leave.
 */
class Slot : public PrimOp {
private:
    Slot(Type type, Def frame, size_t index, const std::string& name);

public:
    Def frame() const { return op(0); }
    size_t index() const { return index_; }
    PtrType ptr_type() const;

    virtual size_t hash() const { return hash_combine(PrimOp::hash(), index()); }
    virtual bool equal(const PrimOp* other) const {
        return PrimOp::equal(other) ? this->index() == other->as<Slot>()->index() : false;
    }

private:
    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

/**
 * This represents a global variable in the data segment.
 */
class Global : public PrimOp {
private:
    Global(Def init, bool is_mutable, const std::string& name);

public:
    Def init() const { return op(0); }
    bool is_mutable() const { return is_mutable_; }
    Type referenced_type() const; ///< Returns the type referenced by this \p Global's pointer type.

    virtual const char* op_name() const;
    virtual size_t hash() const { return hash_value(gid()); }
    virtual bool equal(const PrimOp* other) const { return this == other; }

private:
    bool is_mutable_;

    friend class World;
};

//------------------------------------------------------------------------------

template<class To> 
using PrimOpMap     = HashMap<const PrimOp*, To>;
using PrimOpSet     = HashSet<const PrimOp*>;
using PrimOp2PrimOp = PrimOpMap<const PrimOp*>;

//------------------------------------------------------------------------------

}

#endif
