#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/def.h"
#include "thorin/enums.h"
#include "thorin/util/hash.h"

namespace thorin {

//------------------------------------------------------------------------------

/// Base class for all @p PrimOp%s.
class PrimOp : public Def {
protected:
    PrimOp(NodeKind kind, const Type* type, Defs args, const Location& loc, const std::string& name)
        : Def(kind, type, args.size(), loc, name)
        , is_outdated_(false)
    {
        for (size_t i = 0, e = size(); i != e; ++i)
            set_op(i, args[i]);
    }

    void set_type(const Type* type) { type_ = type; }

public:
    const Def* out(size_t i) const;
    virtual bool is_outdated() const override { return is_outdated_; }
    virtual const Def* rebuild(Def2Def&) const override;
    const Def* rebuild(World& to, Defs ops, const Type* type) const {
        assert(this->size() == ops.size());
        return vrebuild(to, ops, type);
    }
    const Def* rebuild(Defs ops) const { return rebuild(world(), ops, type()); }
    const Def* rebuild(Defs ops, const Type* type) const { return rebuild(world(), ops, type); }
    virtual bool has_multiple_outs() const { return false; }
    virtual const char* op_name() const;
    virtual std::ostream& stream(std::ostream&) const override;
    std::ostream& stream_assignment(std::ostream&) const;

protected:
    virtual uint64_t vhash() const;
    virtual bool equal(const PrimOp* other) const;
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const = 0;
    /// Is @p def the @p i^th result of a @p T @p PrimOp?
    template<int i, class T> inline static const T* is_out(const Def* def);

private:
    uint64_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }

    mutable uint64_t hash_ = 0;
    mutable uint32_t live_ = 0;
    mutable bool is_outdated_ : 1;

    friend struct PrimOpHash;
    friend struct PrimOpEqual;
    friend class World;
    friend class Cleaner;
    friend void Def::replace(const Def*) const;
};

struct PrimOpHash {
    uint64_t operator() (const PrimOp* o) const { return o->hash(); }
};

struct PrimOpEqual {
    bool operator() (const PrimOp* o1, const PrimOp* o2) const { return o1->equal(o2); }
};

//------------------------------------------------------------------------------

/// Base class for all @p PrimOp%s without operands.
class Literal : public PrimOp {
protected:
    Literal(NodeKind kind, const Type* type, const Location& loc, const std::string& name)
        : PrimOp(kind, type, {}, loc, name)
    {}
};

/// This literal represents 'no value'.
class Bottom : public Literal {
private:
    Bottom(const Type* type, const Location& loc, const std::string& name)
        : Literal(Node_Bottom, type, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Data constructor for a @p PrimTypeNode.
class PrimLit : public Literal {
private:
    PrimLit(World& world, PrimTypeKind kind, Box box, const Location& loc, const std::string& name);

public:
    Box value() const { return box_; }
#define THORIN_ALL_TYPE(T, M) T T##_value() const { return value().get_##T(); }
#include "thorin/tables/primtypetable.h"

    const PrimType* type() const { return Literal::type()->as<PrimType>(); }
    PrimTypeKind primtype_kind() const { return type()->primtype_kind(); }

    std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override;
    virtual bool equal(const PrimOp* other) const override;
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    Box box_;

    friend class World;
};

template<class T>
T primlit_value(const Def* def) {
    static_assert(std::is_integral<T>::value, "only integral types supported");
    auto lit = def->as<PrimLit>();
    switch (lit->primtype_kind()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit->value().get_##T();
#include "thorin/tables/primtypetable.h"
        default: THORIN_UNREACHABLE;
    }
}

template<class T>
T get(ArrayRef<T> array, const Def* def) { return array[primlit_value<size_t>(def)]; }


/// Akin to <tt>cond ? tval : fval</tt>.
class Select : public PrimOp {
private:
    Select(const Def* cond, const Def* tval, const Def* fval, const Location& loc, const std::string& name)
        : PrimOp(Node_Select, tval->type(), {cond, tval, fval}, loc, name)
    {
        assert(is_type_bool(cond->type()));
        assert(tval->type() == fval->type() && "types of both values must be equal");
        assert(!tval->type()->isa<FnType>() && "must not be a function");
    }

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* cond() const { return op(0); }
    const Def* tval() const { return op(1); }
    const Def* fval() const { return op(2); }

    friend class World;
};

/// Base class for all side-effect free binary \p PrimOp%s.
class BinOp : public PrimOp {
protected:
    BinOp(NodeKind kind, const Type* type, const Def* lhs, const Def* rhs, const Location& loc, const std::string& name)
        : PrimOp(kind, type, {lhs, rhs}, loc, name)
    {
        assert(lhs->type() == rhs->type() && "types are not equal");
    }

public:
    const Def* lhs() const { return op(0); }
    const Def* rhs() const { return op(1); }
};

/// One of \p ArithOpKind arithmetic operation.
class ArithOp : public BinOp {
private:
    ArithOp(ArithOpKind kind, const Def* lhs, const Def* rhs, const Location& loc, const std::string& name)
        : BinOp((NodeKind) kind, lhs->type(), lhs, rhs, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    ArithOpKind arithop_kind() const { return (ArithOpKind) kind(); }
    virtual const char* op_name() const override;

    friend class World;
};

/// One of \p CmpKind compare.
class Cmp : public BinOp {
private:
    Cmp(CmpKind kind, const Def* lhs, const Def* rhs, const Location& loc, const std::string& name);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    CmpKind cmp_kind() const { return (CmpKind) kind(); }
    virtual const char* op_name() const override;

    friend class World;
};

/// Base class for @p Bitcast and @p Cast.
class ConvOp : public PrimOp {
protected:
    ConvOp(NodeKind kind, const Def* from, const Type* to, const Location& loc, const std::string& name)
        : PrimOp(kind, to, {from}, loc, name)
    {}

public:
    const Def* from() const { return op(0); }
};

/// Converts <tt>from</tt> to type <tt>to</tt>.
class Cast : public ConvOp {
private:
    Cast(const Type* to, const Def* from, const Location& loc, const std::string& name)
        : ConvOp(Node_Cast, from, to, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Reinterprets the bits of <tt>from</tt> as type <tt>to</tt>.
class Bitcast : public ConvOp {
private:
    Bitcast(const Type* to, const Def* from, const Location& loc, const std::string& name)
        : ConvOp(Node_Bitcast, from, to, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Base class for all aggregate data constructers.
class Aggregate : public PrimOp {
protected:
    Aggregate(NodeKind kind, Defs args, const Location& loc, const std::string& name)
        : PrimOp(kind, nullptr /*set later*/, args, loc, name)
    {}
};

/// Data constructor for a \p DefiniteArrayTypeNode.
class DefiniteArray : public Aggregate {
private:
    DefiniteArray(World& world, const Type* elem, Defs args, const Location& loc, const std::string& name);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const DefiniteArrayType* type() const { return Aggregate::type()->as<DefiniteArrayType>(); }
    const Type* elem_type() const { return type()->elem_type(); }

    friend class World;
};

/// Data constructor for an \p IndefiniteArrayTypeNode.
class IndefiniteArray : public Aggregate {
private:
    IndefiniteArray(World& world, const Type* elem, const Def* dim, const Location& loc, const std::string& name);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const IndefiniteArrayType* type() const { return Aggregate::type()->as<IndefiniteArrayType>(); }
    const Type* elem_type() const { return type()->elem_type(); }

    friend class World;
};

/// Data constructor for a @p TupleTypeNode.
class Tuple : public Aggregate {
private:
    Tuple(World& world, Defs args, const Location& loc, const std::string& name);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const TupleType* type() const { return Aggregate::type()->as<TupleType>(); }

    friend class World;
};

/// Data constructor for a @p StructAppTypeNode.
class StructAgg : public Aggregate {
private:
    StructAgg(const StructAppType* struct_app_type, Defs args, const Location& loc, const std::string& name)
        : Aggregate(Node_StructAgg, args, loc, name)
    {
#ifndef NDEBUG
        assert(struct_app_type->num_elems() == args.size());
        for (size_t i = 0, e = args.size(); i != e; ++i)
            assert(struct_app_type->elem(i) == args[i]->type());
#endif
        set_type(struct_app_type);
    }

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const StructAppType* type() const { return Aggregate::type()->as<StructAppType>(); }

    friend class World;
};

/// Data constructor for a vector type.
class Vector : public Aggregate {
private:
    Vector(World& world, Defs args, const Location& loc, const std::string& name);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Base class for functional @p Insert and @p Extract.
class AggOp : public PrimOp {
protected:
    AggOp(NodeKind kind, const Type* type, Defs args, const Location& loc, const std::string& name)
        : PrimOp(kind, type, args, loc, name)
    {}

public:
    const Def* agg() const { return op(0); }
    const Def* index() const { return op(1); }

    friend class World;
};

/// Extracts from aggregate <tt>agg</tt> the element at position <tt>index</tt>.
class Extract : public AggOp {
private:
    Extract(const Def* agg, const Def* index, const Location& loc, const std::string& name)
        : AggOp(Node_Extract, extracted_type(agg, index), {agg, index}, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    static const Type* extracted_type(const Def* agg, const Def* index);

    friend class World;
};

/**
 * Creates a new aggregate by inserting <tt>value</tt> at position <tt>index</tt> into <tt>agg</tt>.
 * @attention { This is a @em functional insert.
 *              The value <tt>agg</tt> remains untouched.
 *              The \p Insert itself is a \em new aggregate which contains the newly created <tt>value</tt>. }
 */
class Insert : public AggOp {
private:
    Insert(const Def* agg, const Def* index, const Def* value, const Location& loc, const std::string& name)
        : AggOp(Node_Insert, agg->type(), {agg, index, value}, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* value() const { return op(2); }

    friend class World;
};

/**
 * Load effective address.
 * Takes a pointer <tt>ptr</tt> to an aggregate as input.
 * Then, the address to the <tt>index</tt>'th element is computed.
 * This yields a pointer to that element.
 */
class LEA : public PrimOp {
private:
    LEA(const Def* ptr, const Def* index, const Location& loc, const std::string& name);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const PtrType* ptr_type() const { return ptr()->type()->as<PtrType>(); }           ///< Returns the PtrType from @p ptr().
    const Type* ptr_referenced_type() const { return ptr_type()->referenced_type(); }  ///< Returns the type referenced by @p ptr().

    friend class World;
};

/// Base class for \p Run and \p Hlt.
class EvalOp : public PrimOp {
protected:
    EvalOp(NodeKind kind, const Def* def, const Location& loc, const std::string& name)
        : PrimOp(kind, def->type(), {def}, loc, name)
    {}

public:
    const Def* def() const { return op(0); }
};

/// Starts a partial evaluation run.
class Run : public EvalOp {
private:
    Run(const Def* def, const Location& loc, const std::string& name)
        : EvalOp(Node_Run, def, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Stops a partial evaluation run or hinders partial evaluation from specializing <tt>def</tt>.
class Hlt : public EvalOp {
private:
    Hlt(const Def* def, const Location& loc, const std::string& name)
        : EvalOp(Node_Hlt, def, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/**
 * A slot in a stack frame opend via @p Enter.
 * A @p Slot yields a pointer to the given <tt>type</tt>.
 * Loads from this address yield @p Bottom if the frame has already been closed.
 */
class Slot : public PrimOp {
private:
    Slot(const Type* type, const Def* frame, size_t index, const Location& loc, const std::string& name);

public:
    const Def* frame() const { return op(0); }
    size_t index() const { return index_; }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Type* alloced_type() const { return type()->referenced_type(); }

private:
    virtual uint64_t vhash() const override;
    virtual bool equal(const PrimOp* other) const override;
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    size_t index_;

    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 */
class Global : public PrimOp {
private:
    Global(const Def* init, bool is_mutable, const Location& loc, const std::string& name);

public:
    const Def* init() const { return op(0); }
    bool is_mutable() const { return is_mutable_; }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Type* alloced_type() const { return type()->referenced_type(); }
    virtual const char* op_name() const override;

    std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override { return hash_value(gid()); }
    virtual bool equal(const PrimOp* other) const override { return this == other; }
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    bool is_mutable_;

    friend class World;
};

/// Base class for all \p PrimOp%s taking and producing side-effects.
class MemOp : public PrimOp {
protected:
    MemOp(NodeKind kind, const Type* type, Defs args, const Location& loc, const std::string& name)
        : PrimOp(kind, type, args, loc, name)
    {
        assert(mem()->type()->isa<MemType>());
        assert(args.size() >= 1);
    }

public:
    const Def* mem() const { return op(0); }
    const Def* out_mem() const { return has_multiple_outs() ? out(0) : this; }

private:
    virtual uint64_t vhash() const override { return hash_value(gid()); }
    virtual bool equal(const PrimOp* other) const override { return this == other; }
};

/// Allocates memory on the heap.
class Alloc : public MemOp {
private:
    Alloc(const Type* type, const Def* mem, const Def* extra, const Location& loc, const std::string& name);

public:
    const Def* extra() const { return op(1); }
    virtual bool has_multiple_outs() const override { return true; }
    const Def* out_ptr() const { return out(1); }
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    const PtrType* out_ptr_type() const { return type()->arg(1)->as<PtrType>(); }
    const Type* alloced_type() const { return out_ptr_type()->referenced_type(); }
    static const Alloc* is_out_mem(const Def* def) { return is_out<0, Alloc>(def); }
    static const Alloc* is_out_ptr(const Def* def) { return is_out<1, Alloc>(def); }

private:
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Base class for @p Load and @p Store.
class Access : public MemOp {
protected:
    Access(NodeKind kind, const Type* type, Defs args, const Location& loc, const std::string& name)
        : MemOp(kind, type, args, loc, name)
    {
        assert(args.size() >= 2);
    }

public:
    const Def* ptr() const { return op(1); }
};

/// Loads with current effect <tt>mem</tt> from <tt>ptr</tt> to produce a pair of a new effect and the loaded value.
class Load : public Access {
private:
    Load(const Def* mem, const Def* ptr, const Location& loc, const std::string& name);

public:
    virtual bool has_multiple_outs() const override { return true; }
    const Def* out_val() const { return out(1); }
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    const Type* out_val_type() const { return type()->arg(1); }
    static const Load* is_out_mem(const Def* def) { return is_out<0, Load>(def); }
    static const Load* is_out_val(const Def* def) { return is_out<1, Load>(def); }

private:
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Stores with current effect <tt>mem</tt> <tt>value</tt> into <tt>ptr</tt> while producing a new effect.
class Store : public Access {
private:
    Store(const Def* mem, const Def* ptr, const Def* value, const Location& loc, const std::string& name)
        : Access(Node_Store, mem->type(), {mem, ptr, value}, loc, name)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* val() const { return op(2); }
    const MemType* type() const { return Access::type()->as<MemType>(); }

    friend class World;
};

/// Creates a stack \p Frame with current effect <tt>mem</tt>.
class Enter : public MemOp {
private:
    Enter(const Def* mem, const Location& loc, const std::string& name);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    virtual bool has_multiple_outs() const override { return true; }
    const Def* out_frame() const { return out(1); }
    static const Enter* is_out_mem(const Def* def) { return is_out<0, Enter>(def); }
    static const Enter* is_out_frame(const Def* def) { return is_out<1, Enter>(def); }

    friend class World;
};

//------------------------------------------------------------------------------

template<int i, class T>
const T* PrimOp::is_out(const Def* def) {
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
