#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/def.h"
#include "thorin/enums.h"
#include "thorin/util/hash.h"

namespace thorin {

template<int i, class T> const T* is_out(Def def);

//------------------------------------------------------------------------------

class PrimOp : public DefNode {
protected:
    PrimOp(NodeKind kind, Type type, ArrayRef<Def> args, const std::string& name)
        : DefNode(-1, kind, type ? type.unify() : nullptr, args.size(), name)
        , up_to_date_(true)
    {
        for (size_t i = 0, e = size(); i != e; ++i)
            set_op(i, args[i]);
    }

    void set_type(Type type) { type_ = type.unify(); }

public:
    bool up_to_date() const { return up_to_date_; }
    virtual Def rebuild() const override;
    Def rebuild(World& to, ArrayRef<Def> ops, Type type) const {
        assert(this->size() == ops.size());
        return vrebuild(to, ops, type);
    }
    Def rebuild(ArrayRef<Def> ops) const { return rebuild(world(), ops, type()); }
    Def rebuild(ArrayRef<Def> ops, Type type) const { return rebuild(world(), ops, type); }
    virtual const char* op_name() const;

protected:
    virtual size_t vhash() const;
    virtual bool equal(const PrimOp* other) const;
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const = 0;

private:
    size_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }
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

    PrimType primtype() const { return type().as<PrimType>(); }
    PrimTypeKind primtype_kind() const { return primtype()->primtype_kind(); }

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
    }

public:
    Def tval() const { return op(1); }
    Def fval() const { return op(2); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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

public:
    ArithOpKind arithop_kind() const { return (ArithOpKind) kind(); }
    virtual const char* op_name() const override;

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Cmp : public BinOp {
private:
    Cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const std::string& name);

public:
    CmpKind cmp_kind() const { return (CmpKind) kind(); }
    virtual const char* op_name() const override;

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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

public:
    DefiniteArrayType type() const { return Aggregate::type().as<DefiniteArrayType>(); }
    Type elem_type() const { return type()->elem_type(); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class IndefiniteArray : public Aggregate {
private:
    IndefiniteArray(World& world, Type elem, Def dim, const std::string& name);

public:
    IndefiniteArrayType type() const { return Aggregate::type().as<IndefiniteArrayType>(); }
    Type elem_type() const { return type()->elem_type(); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Tuple : public Aggregate {
private:
    Tuple(World& world, ArrayRef<Def> args, const std::string& name);

public:
    TupleType tuple_type() const { return Aggregate::type().as<TupleType>(); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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

public:
    static Type determine_type(Def agg, Def index);

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Insert : public AggOp {
private:
    Insert(Def agg, Def index, Def value, const std::string& name)
        : AggOp(Node_Insert, agg->type(), {agg, index, value}, name)
    {}

public:
    Def value() const { return op(2); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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
    /// Returns the PtrType from \p ptr().
    PtrType ptr_type() const { return ptr()->type().as<PtrType>(); }
    /// Returns the type referenced by \p ptr().
    Type referenced_type() const { return ptr_type()->referenced_type(); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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

public:
    Def run() const { return op(1); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class EndHlt : public EndEvalOp {
private:
    EndHlt(Def def, Def hlt, const std::string& name)
        : EndEvalOp(Node_EndHlt, def, hlt, name)
    {}

public:
    Def hlt() const { return op(1); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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
    PtrType type() const { return PrimOp::type().as<PtrType>(); }
    Type alloced_type() const { return type()->referenced_type(); }

private:
    virtual size_t vhash() const override;
    virtual bool equal(const PrimOp* other) const override;
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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
    virtual Def out_mem() const = 0;
};

class Alloc : public MemOp {
private:
    Alloc(Type type, Def mem, Def extra, const std::string& name);

public:
    Def extra() const { return op(1); }
    PtrType alloced_ptr_type() const { return type().as<TupleType>()->arg(1).as<PtrType>(); }
    Type alloced_referenced_type() const { return alloced_ptr_type()->referenced_type(); }
    Def extract_mem() const;
    Def extract_ptr() const;
    static const Alloc* is_mem(Def def) { return is_out<0, Alloc>(def); }
    static const Alloc* is_ptr(Def def) { return is_out<1, Alloc>(def); }
    virtual Def out_mem() const override { return extract_mem(); }

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
    Def extract_mem() const;
    Def extract_val() const;
    static const Load* is_mem(Def def) { return is_out<0, Load>(def); }
    static const Load* is_val(Def def) { return is_out<1, Load>(def); }
    virtual Def out_mem() const override { return extract_mem(); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Store : public Access {
private:
    Store(Def mem, Def ptr, Def value, const std::string& name)
        : Access(Node_Store, mem->type(), {mem, ptr, value}, name)
    {}

public:
    Def val() const { return op(2); }
    virtual Def out_mem() const override { return this; }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Enter : public MemOp {
private:
    Enter(Def mem, const std::string& name);

public:
    Def extract_mem() const;
    Def extract_frame() const;
    static const Enter* is_mem(Def def) { return is_out<0, Enter>(def); }
    static const Enter* is_ptr(Def def) { return is_out<1, Enter>(def); }
    virtual Def out_mem() const override { return extract_mem(); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class MapOp : public Access {
protected:
    MapOp(NodeKind kind, Type type, ArrayRef<Def> args, const std::string& name)
        : Access(kind, type, args, name)
    {}
};

class Map : public MapOp {
private:
    Map(int32_t device, AddressSpace addr_space, Def mem, Def ptr, Def offset, Def size, const std::string& name);

public:
    Def mem_offset() const { return op(2); }
    Def mem_size() const { return op(3); }
    PtrType ptr_type() const { return type().as<TupleType>()->arg(1).as<PtrType>(); }
    AddressSpace addr_space() const { return ptr_type()->addr_space(); }
    int32_t device() const { return ptr_type()->device(); }
    Def extract_mem() const;
    Def extract_ptr() const;
    static const Map* is_mem(Def def) { return is_out<0, Map>(def); }
    static const Map* is_ptr(Def def) { return is_out<1, Map>(def); }
    virtual Def out_mem() const override { return extract_mem(); }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

    friend class World;
};

class Unmap : public MapOp {
private:
    Unmap(Def mem, Def ptr, const std::string& name)
        : MapOp(Node_Unmap, mem->type(), {mem, ptr}, name)
    {}

public:
    virtual Def out_mem() const override { return this; }

private:
    virtual Def vrebuild(World& to, ArrayRef<Def> ops, Type type) const override;

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

/// Is \p def the \p i^th result of a \p T \p PrimOp?
template<int i, class T>
const T* is_out(Def def) {
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
