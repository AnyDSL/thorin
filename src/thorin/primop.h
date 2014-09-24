#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/def.h"
#include "thorin/enums.h"
#include "thorin/util/hash.h"

namespace thorin {

//------------------------------------------------------------------------------

class PrimOp : public DefNode {
protected:
    PrimOp(size_t size, NodeKind kind, Type type, const std::string& name)
        : DefNode(-1, kind, size, type ? type.unify() : nullptr, name)
        , up_to_date_(true)
    {}

    void set_type(Type type) { type_ = type.unify(); }

public:
    bool up_to_date() const { return up_to_date_; }
    virtual Def rebuild() const override;
    virtual const char* op_name() const;
    size_t hash() const {
        if (hash_ == 0)
            hash_ = vhash();
        return hash_;
    }
    virtual size_t vhash() const {
        size_t seed = hash_combine(hash_combine(hash_begin((int) kind()), size()), type().unify()->gid());
        for (auto op : ops_)
            seed = hash_combine(seed, op.node()->gid());
        return seed;
    }
    virtual bool equal(const PrimOp* other) const {
        bool result = this->kind() == other->kind() && this->size() == other->size() && this->type() == other->type();
        for (size_t i = 0, e = size(); result && i != e; ++i)
            result &= this->ops_[i].node() == other->ops_[i].node();
        return result;
    }

private:
    void set_gid(size_t gid) const { const_cast<size_t&>(const_cast<PrimOp*>(this)->gid_) = gid; }

    mutable size_t hash_ = 0;
    mutable uint32_t live_ = 0;
    mutable bool up_to_date_ : 1;

    friend struct PrimOpHash;
    friend struct PrimOpEqual;
    friend class World;
    friend class Cleaner;
    friend void DefNode::replace(Def) const;
};

struct PrimOpHash { size_t operator () (const PrimOp* o) const { return o->hash(); } };
struct PrimOpEqual { bool operator () (const PrimOp* o1, const PrimOp* o2) const { return o1->equal(o2); } };

//------------------------------------------------------------------------------

class Literal : public PrimOp {
protected:
    Literal(NodeKind kind, Type type, const std::string& name)
        : PrimOp(0, kind, type, name)
    {}
};

/// Base class for \p Any and \p Bottom.
class Undef : public Literal {
protected:
    Undef(NodeKind kind, Type type, const std::string& name)
        : Literal(kind, type, name)
    {}
};

/**
 * @brief The wish-you-a-value value.
 *
 * This literal represents an arbitrary value.
 * When ever an operation takes an \p Undef value as argument,
 * you may literally wish your favorite value instead.
 */
class Any : public Undef {
private:
    Any(Type type, const std::string& name)
        : Undef(Node_Any, type, name)
    {}

    friend class World;
};

/**
 * @brief The novalue-value.
 *
 * This literal represents literally 'no value'.
 * Extremely useful for data flow analysis.
 */
class Bottom : public Undef {
private:
    Bottom(Type type, const std::string& name)
        : Undef(Node_Bottom, type, name)
    {}

    friend class World;
};

class PrimLit : public Literal {
private:
    PrimLit(World& world, PrimTypeKind kind, Box box, const std::string& name);

public:
    Box value() const { return box_; }
#define THORIN_ALL_TYPE(T, M) T T##_value() const { return value().get_##T(); }
#include "thorin/tables/primtypetable.h"

    PrimType primtype() const { return type().as<PrimType>(); }
    PrimTypeKind primtype_kind() const { return primtype()->primtype_kind(); }
    virtual size_t vhash() const override { return hash_combine(Literal::vhash(), bcast<uint64_t, Box>(value())); }
    virtual bool equal(const PrimOp* other) const override {
        return Literal::equal(other) ? this->value() == other->as<PrimLit>()->value() : false;
    }

private:
    Box box_;

    friend class World;
};

class VectorOp : public PrimOp {
protected:
    VectorOp(size_t size, NodeKind kind, Type type, Def cond, const std::string& name)
        : PrimOp(size, kind, type, name)
    {
        assert(cond->type()->is_bool());
        set_op(0, cond);
    }

public:
    Def cond() const { return op(0); }
};

class Select : public VectorOp {
private:
    Select(Def cond, Def tval, Def fval, const std::string& name)
        : VectorOp(3, Node_Select, tval->type(), cond, name)
    {
        set_op(1, tval);
        set_op(2, fval);
        assert(tval->type() == fval->type() && "types of both values must be equal");
    }

public:
    Def tval() const { return op(1); }
    Def fval() const { return op(2); }

    friend class World;
};

class BinOp : public VectorOp {
protected:
    BinOp(NodeKind kind, Type type, Def cond, Def lhs, Def rhs, const std::string& name)
        : VectorOp(3, kind, type, cond, name)
    {
        assert(lhs->type() == rhs->type() && "types are not equal");
        set_op(1, lhs);
        set_op(2, rhs);
    }

public:
    Def lhs() const { return op(1); }
    Def rhs() const { return op(2); }
};

class ArithOp : public BinOp {
private:
    ArithOp(ArithOpKind kind, Def cond, Def lhs, Def rhs, const std::string& name)
        : BinOp((NodeKind) kind, lhs->type(), cond, lhs, rhs, name)
    {}

public:
    ArithOpKind arithop_kind() const { return (ArithOpKind) kind(); }
    virtual const char* op_name() const override;

    friend class World;
};

class Cmp : public BinOp {
private:
    Cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const std::string& name);

public:
    CmpKind cmp_kind() const { return (CmpKind) kind(); }
    virtual const char* op_name() const override;

    friend class World;
};

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
    Type elem_type() const { return type()->elem_type(); }

    friend class World;
};

class IndefiniteArray : public Aggregate {
private:
    IndefiniteArray(World& world, Type elem, Def dim, const std::string& name);

public:
    IndefiniteArrayType type() const { return Aggregate::type().as<IndefiniteArrayType>(); }
    Type elem_type() const { return type()->elem_type(); }

    friend class World;
};

class Tuple : public Aggregate {
private:
    Tuple(World& world, ArrayRef<Def> args, const std::string& name);

public:
    TupleType tuple_type() const { return Aggregate::type().as<TupleType>(); }

    friend class World;
};

class StructAgg : public Aggregate {
private:
    StructAgg(StructAppType struct_app_type, ArrayRef<Def> args, const std::string& name)
        : Aggregate(Node_StructAgg, args, name)
    {
#ifndef NDEBUG
        assert(struct_app_type->num_elems() == args.size());
        for (size_t i = 0, e = args.size(); i != e; ++i)
            assert(struct_app_type->elem(i) == args[i]->type());
#endif

        set_type(struct_app_type);
    }

public:
    StructAppType struct_app_type() const { return Aggregate::type().as<StructAppType>(); }

    friend class World;
};

class Vector : public Aggregate {
private:
    Vector(World& world, ArrayRef<Def> args, const std::string& name);
    friend class World;
};

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
    Extract(Def agg, Def index, const std::string& name)
        : AggOp(2, Node_Extract, type(agg, index), agg, index, name)
    {}

public:
    static Type type(Def agg, Def index);

    friend class World;
};

class Insert : public AggOp {
private:
    Insert(Def agg, Def index, Def value, const std::string& name)
        : AggOp(3, Node_Insert, agg->type(), agg, index, name)
    {
        set_op(2, value);
    }

public:
    Def value() const { return op(2); }

    friend class World;
};

/**
 * Load Effective Address.
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

class EvalOp : public PrimOp {
protected:
    EvalOp(NodeKind kind, Def def, const std::string& name)
        : PrimOp(1, kind, def->type(), name)
    {
        set_op(0, def);
    }

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

class EndEvalOp : public PrimOp {
protected:
    EndEvalOp(NodeKind kind, Def def, Def eval, const std::string& name)
        : PrimOp(2, kind, def->type(), name)
    {
        set_op(0, def);
        set_op(1, eval);
    }

public:
    Def def() const { return op(0); }
    Def eval() const { return op(1); }
};

class EndRun : public EndEvalOp {
private:
    EndRun(Def def, Def run, const std::string& name)
        : EndEvalOp(Node_EndRun, def, run, name)
    {}

public:
    Def run() const { return op(1); }

    friend class World;
};

class EndHlt : public EndEvalOp {
private:
    EndHlt(Def def, Def hlt, const std::string& name)
        : EndEvalOp(Node_EndHlt, def, hlt, name)
    {}

public:
    Def hlt() const { return op(1); }

    friend class World;
};

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

    virtual size_t vhash() const override { return hash_combine(PrimOp::vhash(), index()); }
    virtual bool equal(const PrimOp* other) const override {
        return PrimOp::equal(other) ? this->index() == other->as<Slot>()->index() : false;
    }

private:
    size_t index_;

    friend class World;
};

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

    virtual const char* op_name() const override;
    virtual size_t vhash() const override { return hash_value(gid()); }
    virtual bool equal(const PrimOp* other) const override { return this == other; }

private:
    bool is_mutable_;

    friend class World;
};

class MemOp : public PrimOp {
protected:
    MemOp(size_t size, NodeKind kind, Type type, Def mem, const std::string& name)
        : PrimOp(size, kind, type, name)
    {
        assert(mem->type().isa<MemType>());
        assert(size >= 1);
        set_op(0, mem);
    }

public:
    Def mem() const { return op(0); }
    virtual bool has_mem_out() const { return false; }
    virtual Def mem_out() const { return Def(); }
};

class Alloc : public MemOp {
private:
    Alloc(Def mem, Type type, Def extra, const std::string& name);

public:
    Def extra() const { return op(1); }
    Type alloced_type() const { return type().as<PtrType>()->referenced_type(); }

    friend class World;
};

class Access : public MemOp {
protected:
    Access(size_t size, NodeKind kind, Type type, Def mem, Def ptr, const std::string& name)
        : MemOp(size, kind, type, mem, name)
    {
        assert(size >= 2);
        set_op(1, ptr);
    }

public:
    Def ptr() const { return op(1); }
};

class Load : public Access {
private:
    Load(Def mem, Def ptr, const std::string& name)
        : Access(2, Node_Load, ptr->type().as<PtrType>()->referenced_type(), mem, ptr, name)
    {}

public:
    Def ptr() const { return op(1); }

    friend class World;
};

class Store : public Access {
private:
    Store(Def mem, Def ptr, Def value, const std::string& name)
        : Access(3, Node_Store, mem->type(), mem, ptr, name)
    {
        set_op(2, value);
    }

public:
    Def val() const { return op(2); }
    virtual bool has_mem_out() const override { return true; }
    virtual Def mem_out() const override { return this; }

    friend class World;
};

class Enter : public MemOp {
private:
    Enter(Def mem, const std::string& name);

    friend class World;
};

class MapOp : public MemOp {
protected:
    MapOp(size_t size, NodeKind kind, Type type, Def mem, Def ptr, const std::string &name)
        : MemOp(size, kind, type, mem, name)
    {
        set_op(1, ptr);
    }

public:
    Def ptr() const { return op(1); }
};

class Map : public MapOp {
private:
    Map(Def mem, Def ptr, int32_t device, AddressSpace addr_space,
        Def offset, Def size, const std::string &name);

public:
    Def extract_mem() const;
    Def extract_mapped_ptr() const;
    Def mem_offset() const { return op(2); }
    Def mem_size() const { return op(3); }
    PtrType ptr_type() const { return type().as<TupleType>()->arg(1).as<PtrType>(); }
    AddressSpace addr_space() const { return ptr_type()->addr_space(); }
    int32_t device() const { return ptr_type()->device(); }
    virtual bool has_mem_out() const override { return true; }
    virtual Def mem_out() const override;

    friend class World;
};

class Unmap : public MapOp {
private:
    Unmap(Def mem, Def ptr, const std::string &name)
        : MapOp(2, Node_Unmap, mem->type(), mem, ptr, name)
    {}

    virtual bool has_mem_out() const override { return true; }
    virtual Def mem_out() const override { return this; }

    friend class World;
};

//------------------------------------------------------------------------------

template<class T>
T DefNode::primlit_value() const {
    const PrimLit* lit = this->as<PrimLit>();
    switch (lit->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit->value().get_##T();
#include "thorin/tables/primtypetable.h"
        default: THORIN_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

template<class To>
using PrimOpMap     = HashMap<const PrimOp*, To>;
using PrimOpSet     = HashSet<const PrimOp*>;
using PrimOp2PrimOp = PrimOpMap<const PrimOp*>;

//------------------------------------------------------------------------------

}

#endif
