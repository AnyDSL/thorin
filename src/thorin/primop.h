#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/def.h"
#include "thorin/enums.h"
#include "thorin/util/hash.h"

namespace thorin {

//------------------------------------------------------------------------------

class PrimOp : public DefNode {
protected:
    PrimOp(NodeKind kind, Type type, ArrayRef<Def> args, const std::string& name)
        : DefNode(-1, kind, type ? type.unify() : nullptr, args.size(), name)
        , is_outdated_(false)
    {
        for (size_t i = 0, e = size(); i != e; ++i)
            set_op(i, args[i]);
    }

    void set_type(Type type) { type_ = type.unify(); }

public:
    Def out(size_t i) const;
    bool is_outdated() const { return is_outdated_; }
    virtual Def rebuild() const override;
    Def rebuild(World& to, ArrayRef<Def> ops, Type type) const {
        assert(this->size() == ops.size());
        return vrebuild(to, ops, type);
    }
    Def rebuild(ArrayRef<Def> ops) const { return rebuild(world(), ops, type()); }
    Def rebuild(ArrayRef<Def> ops, Type type) const { return rebuild(world(), ops, type); }
    virtual bool has_multiple_outs() const { return false; }
    virtual const char* op_name() const;

protected:
    virtual size_t vhash() const;
    virtual bool equal(const PrimOp* other) const;
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const = 0;
    /// Is \p def the \p i^th result of a \p T \p PrimOp?
    template<int i, class T> inline static const T* is_out(Def def);

private:
    size_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }
    void set_gid(size_t gid) const { const_cast<size_t&>(const_cast<PrimOp*>(this)->gid_) = gid; }

    mutable size_t hash_ = 0;
    mutable uint32_t live_ = 0;
    mutable bool is_outdated_ : 1;

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
        : PrimOp(kind, type, {}, name)
    {}
};

/// This literal represents 'no value'.
class Bottom : public Literal {
private:
    Bottom(Type type, const std::string& name)
        : Literal(Node_Bottom, type, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class PrimLit : public Literal {
private:
    PrimLit(World& world, PrimTypeKind kind, Box box, const std::string& name);

public:
    Box value() const { return box_; }
#define THORIN_ALL_TYPE(T, M) T T##_value() const { return value().get_##T(); }
#include "thorin/tables/primtypetable.h"

    PrimType type() const { return Literal::type().as<PrimType>(); }
    PrimTypeKind primtype_kind() const { return type()->primtype_kind(); }

private:
    virtual size_t vhash() const override;
    virtual bool equal(const PrimOp* other) const override;
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    Box box_;

    friend class World;
};

class VectorOp : public PrimOp {
protected:
    VectorOp(NodeKind kind, Type type, ArrayRef<Def> args, const std::string& name)
        : PrimOp(kind, type, args, name)
    {
        assert(cond()->type()->is_bool());
    }

public:
    Def cond() const { return op(0); }
};

class Select : public VectorOp {
private:
    Select(Def cond, Def tval, Def fval, const std::string& name)
        : VectorOp(Node_Select, tval->type(), {cond, tval, fval}, name)
    {
        assert(tval->type() == fval->type() && "types of both values must be equal");
        assert(!tval->type().isa<FnType>() && "must not be a function");
    }

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    Def tval() const { return op(1); }
    Def fval() const { return op(2); }

    friend class World;
};

class BinOp : public VectorOp {
protected:
    BinOp(NodeKind kind, Type type, Def cond, Def lhs, Def rhs, const std::string& name)
        : VectorOp(kind, type, {cond, lhs, rhs}, name)
    {
        assert(lhs->type() == rhs->type() && "types are not equal");
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

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    PrimType type() const { return BinOp::type().as<PrimType>(); }
    ArithOpKind arithop_kind() const { return (ArithOpKind) kind(); }
    virtual const char* op_name() const override;

    friend class World;
};

class Cmp : public BinOp {
private:
    Cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const std::string& name);

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    PrimType type() const { return BinOp::type().as<PrimType>(); }
    CmpKind cmp_kind() const { return (CmpKind) kind(); }
    virtual const char* op_name() const override;

    friend class World;
};

class ConvOp : public VectorOp {
protected:
    ConvOp(NodeKind kind, Def cond, Def from, Type to, const std::string& name)
        : VectorOp(kind, to, {cond, from}, name)
    {}

public:
    Def from() const { return op(1); }
};

class Cast : public ConvOp {
private:
    Cast(Type to, Def cond, Def from, const std::string& name)
        : ConvOp(Node_Cast, cond, from, to, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Bitcast : public ConvOp {
private:
    Bitcast(Type to, Def cond, Def from, const std::string& name)
        : ConvOp(Node_Bitcast, cond, from, to, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Aggregate : public PrimOp {
protected:
    Aggregate(NodeKind kind, ArrayRef<Def> args, const std::string& name)
        : PrimOp(kind, Type() /*set later*/, args, name)
    {}
};

class DefiniteArray : public Aggregate {
private:
    DefiniteArray(World& world, Type elem, ArrayRef<Def> args, const std::string& name);

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    DefiniteArrayType type() const { return Aggregate::type().as<DefiniteArrayType>(); }
    Type elem_type() const { return type()->elem_type(); }

    friend class World;
};

class IndefiniteArray : public Aggregate {
private:
    IndefiniteArray(World& world, Type elem, Def dim, const std::string& name);

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    IndefiniteArrayType type() const { return Aggregate::type().as<IndefiniteArrayType>(); }
    Type elem_type() const { return type()->elem_type(); }

    friend class World;
};

class Tuple : public Aggregate {
private:
    Tuple(World& world, ArrayRef<Def> args, const std::string& name);

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    TupleType type() const { return Aggregate::type().as<TupleType>(); }

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

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    StructAppType type() const { return Aggregate::type().as<StructAppType>(); }

    friend class World;
};

class Vector : public Aggregate {
private:
    Vector(World& world, ArrayRef<Def> args, const std::string& name);

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class AggOp : public PrimOp {
protected:
    AggOp(NodeKind kind, Type type, ArrayRef<Def> args, const std::string& name)
        : PrimOp(kind, type, args, name)
    {}

public:
    Def agg() const { return op(0); }
    Def index() const { return op(1); }

    friend class World;
};

class Extract : public AggOp {
private:
    Extract(Def agg, Def index, const std::string& name)
        : AggOp(Node_Extract, determine_type(agg, index), {agg, index}, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    static Type determine_type(Def agg, Def index);

    friend class World;
};

class Insert : public AggOp {
private:
    Insert(Def agg, Def index, Def value, const std::string& name)
        : AggOp(Node_Insert, agg->type(), {agg, index, value}, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    Def ptr() const { return op(0); }
    Def index() const { return op(1); }
    PtrType type() const { return PrimOp::type().as<PtrType>(); }
    PtrType ptr_type() const { return ptr()->type().as<PtrType>(); }            ///< Returns the PtrType from \p ptr().
    Type ptr_referenced_type() const { return ptr_type()->referenced_type(); }  ///< Returns the type referenced by \p ptr().

    friend class World;
};

class EvalOp : public PrimOp {
protected:
    EvalOp(NodeKind kind, Def def, const std::string& name)
        : PrimOp(kind, def->type(), {def}, name)
    {}

public:
    Def def() const { return op(0); }
};

class Run : public EvalOp {
private:
    Run(Def def, const std::string& name)
        : EvalOp(Node_Run, def, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Hlt : public EvalOp {
private:
    Hlt(Def def, const std::string& name)
        : EvalOp(Node_Hlt, def, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class EndEvalOp : public PrimOp {
protected:
    EndEvalOp(NodeKind kind, Def def, Def eval, const std::string& name)
        : PrimOp(kind, def->type(), {def, eval}, name)
    {}

public:
    Def def() const { return op(0); }
    Def eval() const { return op(1); }
};

class EndRun : public EndEvalOp {
private:
    EndRun(Def def, Def run, const std::string& name)
        : EndEvalOp(Node_EndRun, def, run, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    Def run() const { return op(1); }

    friend class World;
};

class EndHlt : public EndEvalOp {
private:
    EndHlt(Def def, Def hlt, const std::string& name)
        : EndEvalOp(Node_EndHlt, def, hlt, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    Def hlt() const { return op(1); }

    friend class World;
};

/// This represents a slot in a stack frame opend via \p Enter.
/// Loads from this address yield \p Bottom if the frame has already been closed via \p Leave.
class Slot : public PrimOp {
private:
    Slot(Type type, Def frame, size_t index, const std::string& name);

public:
    Def frame() const { return op(0); }
    size_t index() const { return index_; }
    PtrType type() const { return PrimOp::type().as<PtrType>(); }
    Type alloced_type() const { return type()->referenced_type(); }

private:
    virtual size_t vhash() const override;
    virtual bool equal(const PrimOp* other) const override;
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    size_t index_;

    friend class World;
};

/// This represents a global variable in the data segment.
class Global : public PrimOp {
private:
    Global(Def init, bool is_mutable, const std::string& name);

public:
    Def init() const { return op(0); }
    bool is_mutable() const { return is_mutable_; }
    PtrType type() const { return PrimOp::type().as<PtrType>(); }
    Type alloced_type() const { return type()->referenced_type(); }
    virtual const char* op_name() const override;

private:
    virtual size_t vhash() const override { return hash_value(gid()); }
    virtual bool equal(const PrimOp* other) const override { return this == other; }
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    bool is_mutable_;

    friend class World;
};

class MemOp : public PrimOp {
protected:
    MemOp(NodeKind kind, Type type, ArrayRef<Def> args, const std::string& name)
        : PrimOp(kind, type, args, name)
    {
        assert(mem()->type().isa<MemType>());
        assert(args.size() >= 1);
    }

public:
    Def mem() const { return op(0); }
    Def out_mem() const { return has_multiple_outs() ? out(0) : this; }

private:
    virtual size_t vhash() const override { return hash_value(gid()); }
    virtual bool equal(const PrimOp* other) const override { return this == other; }
};

class Alloc : public MemOp {
private:
    Alloc(Type type, Def mem, Def extra, const std::string& name);

public:
    Def extra() const { return op(1); }
    virtual bool has_multiple_outs() const { return true; }
    Def out_ptr() const { return out(1); }
    TupleType type() const { return MemOp::type().as<TupleType>(); }
    PtrType out_ptr_type() const { return type()->arg(1).as<PtrType>(); }
    Type alloced_type() const { return out_ptr_type()->referenced_type(); }
    static const Alloc* is_out_mem(Def def) { return is_out<0, Alloc>(def); }
    static const Alloc* is_out_ptr(Def def) { return is_out<1, Alloc>(def); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Access : public MemOp {
protected:
    Access(NodeKind kind, Type type, ArrayRef<Def> args, const std::string& name)
        : MemOp(kind, type, args, name)
    {
        assert(args.size() >= 2);
    }

public:
    Def ptr() const { return op(1); }
};

class Load : public Access {
private:
    Load(Def mem, Def ptr, const std::string& name);

public:
    virtual bool has_multiple_outs() const { return true; }
    Def out_val() const { return out(1); }
    TupleType type() const { return MemOp::type().as<TupleType>(); }
    Type out_val_type() const { return type()->arg(1); }
    static const Load* is_out_mem(Def def) { return is_out<0, Load>(def); }
    static const Load* is_out_val(Def def) { return is_out<1, Load>(def); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Store : public Access {
private:
    Store(Def mem, Def ptr, Def value, const std::string& name)
        : Access(Node_Store, mem->type(), {mem, ptr, value}, name)
    {}

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    Def val() const { return op(2); }
    MemType type() const { return type().as<MemType>(); }

    friend class World;
};

class Enter : public MemOp {
private:
    Enter(Def mem, const std::string& name);

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    TupleType type() const { return MemOp::type().as<TupleType>(); }
    virtual bool has_multiple_outs() const { return true; }
    Def out_frame() const { return out(1); }
    static const Enter* is_out_mem(Def def) { return is_out<0, Enter>(def); }
    static const Enter* is_out_frame(Def def) { return is_out<1, Enter>(def); }

    friend class World;
};

class Map : public Access {
private:
    Map(int32_t device, AddressSpace addr_space, Def mem, Def ptr, Def offset, Def size, const std::string& name);

    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

public:
    Def mem_offset() const { return op(2); }
    Def mem_size() const { return op(3); }
    virtual bool has_multiple_outs() const { return true; }
    Def out_ptr() const { return out(1); }
    TupleType type() const { return Access::type().as<TupleType>(); }
    PtrType out_ptr_type() const { return type()->arg(1).as<PtrType>(); }
    AddressSpace addr_space() const { return out_ptr_type()->addr_space(); }
    int32_t device() const { return out_ptr_type()->device(); }
    static const Map* is_out_mem(Def def) { return is_out<0, Map>(def); }
    static const Map* is_out_ptr(Def def) { return is_out<1, Map>(def); }

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

template<int i, class T>
const T* PrimOp::is_out(Def def) {
    if (auto extract = def->isa<Extract>()) {
        if (extract->index()->is_primlit(i)) {
            if (auto res = extract->agg()->isa<T>())
                return res;
        }
    }
    return nullptr;
}

//------------------------------------------------------------------------------

template<class To>
using PrimOpMap     = HashMap<const PrimOp*, To>;
using PrimOpSet     = HashSet<const PrimOp*>;
using PrimOp2PrimOp = PrimOpMap<const PrimOp*>;

//------------------------------------------------------------------------------

}

#endif
