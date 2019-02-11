#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/enums.h"
#include "thorin/util/hash.h"

namespace thorin {

//------------------------------------------------------------------------------

/// Base class for all @p PrimOp%s.
class PrimOp : public Def {
protected:
    PrimOp(NodeTag tag, const Type* type, Defs args, Debug dbg)
        : Def(tag, type, args.size(), dbg)
    {
        for (size_t i = 0, e = num_ops(); i != e; ++i)
            set_op(i, args[i]);
    }

    void set_type(const Type* type) { type_ = type; }

public:
    const Def* out(size_t i) const;
    const Def* rebuild(World& to, Defs ops, const Type* type) const {
        assert(this->num_ops() == ops.size());
        return vrebuild(to, ops, type);
    }
    const Def* rebuild(Defs ops) const { return rebuild(world(), ops, type()); }
    const Def* rebuild(Defs ops, const Type* type) const { return rebuild(world(), ops, type); }
    virtual bool has_multiple_outs() const { return false; }
    virtual const char* op_name() const;
    virtual std::ostream& stream(std::ostream&) const override;
    virtual std::ostream& stream_assignment(std::ostream&) const;

protected:
    virtual uint64_t vhash() const;
    virtual bool equal(const PrimOp* other) const;
    virtual const Def* vrebuild(World&, Defs, const Type*) const { return nullptr; } //  = 0;

    /// Is @p def the @p i^th result of a @p T @p PrimOp?
    template<int i, class T> inline static const T* is_out(const Def* def);

private:
    uint64_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }

    mutable uint64_t hash_ = 0;

    friend struct PrimOpHash;
    friend class World;
    friend class Cleaner;
    friend void Def::replace(Tracker) const;
};

struct PrimOpHash {
    static uint64_t hash(const PrimOp* o) { return o->hash(); }
    static bool eq(const PrimOp* o1, const PrimOp* o2) { return o1->equal(o2); }
    static const PrimOp* sentinel() { return (const PrimOp*)(1); }
};

//------------------------------------------------------------------------------

/// Base class for all @p PrimOp%s without operands.
class Literal : public PrimOp {
protected:
    Literal(NodeTag tag, const Type* type, Debug dbg)
        : PrimOp(tag, type, {}, dbg)
    {}
};

/// This literal represents 'no value'.
class Bottom : public Literal {
private:
    Bottom(const Type* type, Debug dbg)
        : Literal(Node_Bottom, type, dbg)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// This literal represents 'any value'.
class Top : public Literal {
private:
    Top(const Type* type, Debug dbg)
        : Literal(Node_Top, type, dbg)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Data constructor for a @p PrimType.
class PrimLit : public Literal {
private:
    PrimLit(World& world, PrimTypeTag tag, Box box, Debug dbg);

public:
    Box value() const { return box_; }
#define THORIN_ALL_TYPE(T, M) T T##_value() const { return value().get_##T(); }
#include "thorin/tables/primtypetable.h"

    const PrimType* type() const { return Literal::type()->as<PrimType>(); }
    PrimTypeTag primtype_tag() const { return type()->primtype_tag(); }

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
    switch (lit->primtype_tag()) {
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
    Select(const Def* cond, const Def* tval, const Def* fval, Debug dbg)
        : PrimOp(Node_Select, tval->type(), {cond, tval, fval}, dbg)
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

/// Get number of bytes needed for any value (including bottom) of a given @p Type.
class SizeOf : public PrimOp {
private:
    SizeOf(const Def* def, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Type* of() const { return op(0)->type(); }

    friend class World;
};

/// Base class for all side-effect free binary \p PrimOp%s.
class BinOp : public PrimOp {
protected:
    BinOp(NodeTag tag, const Type* type, const Def* lhs, const Def* rhs, Debug dbg)
        : PrimOp(tag, type, {lhs, rhs}, dbg)
    {
        assert(lhs->type() == rhs->type() && "types are not equal");
    }

public:
    const Def* lhs() const { return op(0); }
    const Def* rhs() const { return op(1); }
};

/// One of \p ArithOpTag arithmetic operation.
class ArithOp : public BinOp {
private:
    ArithOp(ArithOpTag tag, const Def* lhs, const Def* rhs, Debug dbg)
        : BinOp((NodeTag) tag, lhs->type(), lhs, rhs, dbg)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    ArithOpTag arithop_tag() const { return (ArithOpTag) tag(); }
    virtual const char* op_name() const override;

    friend class World;
};

/// One of \p CmpTag compare.
class Cmp : public BinOp {
private:
    Cmp(CmpTag tag, const Def* lhs, const Def* rhs, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    CmpTag cmp_tag() const { return (CmpTag) tag(); }
    virtual const char* op_name() const override;

    friend class World;
};

/// Base class for @p Bitcast and @p Cast.
class ConvOp : public PrimOp {
protected:
    ConvOp(NodeTag tag, const Def* from, const Type* to, Debug dbg)
        : PrimOp(tag, to, {from}, dbg)
    {}

public:
    const Def* from() const { return op(0); }
};

/// Converts <tt>from</tt> to type <tt>to</tt>.
class Cast : public ConvOp {
private:
    Cast(const Type* to, const Def* from, Debug dbg)
        : ConvOp(Node_Cast, from, to, dbg)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Reinterprets the bits of <tt>from</tt> as type <tt>to</tt>.
class Bitcast : public ConvOp {
private:
    Bitcast(const Type* to, const Def* from, Debug dbg)
        : ConvOp(Node_Bitcast, from, to, dbg)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Base class for all aggregate data constructers.
class Aggregate : public PrimOp {
protected:
    Aggregate(NodeTag tag, Defs args, Debug dbg)
        : PrimOp(tag, nullptr /*set later*/, args, dbg)
    {}
};

/// Data constructor for a \p DefiniteArrayType.
class DefiniteArray : public Aggregate {
private:
    DefiniteArray(World& world, const Type* elem, Defs args, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const DefiniteArrayType* type() const { return Aggregate::type()->as<DefiniteArrayType>(); }
    const Type* elem_type() const { return type()->elem_type(); }
    std::string as_string() const;

    friend class World;
};

/// Data constructor for an \p IndefiniteArrayType.
class IndefiniteArray : public Aggregate {
private:
    IndefiniteArray(World& world, const Type* elem, const Def* dim, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const IndefiniteArrayType* type() const { return Aggregate::type()->as<IndefiniteArrayType>(); }
    const Type* elem_type() const { return type()->elem_type(); }

    friend class World;
};

/// Data constructor for a @p TupleType.
class Tuple : public Aggregate {
private:
    Tuple(World& world, Defs args, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const TupleType* type() const { return Aggregate::type()->as<TupleType>(); }

    friend class World;
};

/// Data constructor for a @p ClosureType.
class Closure : public Aggregate {
private:
    Closure(const ClosureType* closure_type, const Def* fn, const Def* env, Debug dbg)
        : Aggregate(Node_Closure, {fn, env}, dbg)
    {
        set_type(closure_type);
    }

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    bool is_thin() const;
    static const VariantType* environment_type(World&);
    static const PtrType*     environment_ptr_type(World&);

    friend class World;
};

/// Data constructor for a @p VariantType.
class Variant : public PrimOp {
private:
    Variant(const VariantType* variant_type, const Def* value, Debug dbg)
        : PrimOp(Node_Variant, variant_type, {value}, dbg)
    {
        assert(std::find(variant_type->ops().begin(), variant_type->ops().end(), value->type()) != variant_type->ops().end());
        set_type(variant_type);
    }

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const VariantType* type() const { return PrimOp::type()->as<VariantType>(); }

    friend class World;
};

/// Data constructor for a @p StructType.
class StructAgg : public Aggregate {
private:
    StructAgg(const StructType* struct_type, Defs args, Debug dbg)
        : Aggregate(Node_StructAgg, args, dbg)
    {
#if THORIN_ENABLE_CHECKS
        assert(struct_type->num_ops() == args.size());
        for (size_t i = 0, e = args.size(); i != e; ++i)
            assert(struct_type->op(i) == args[i]->type());
#endif
        set_type(struct_type);
    }

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const StructType* type() const { return Aggregate::type()->as<StructType>(); }

    friend class World;
};

/// Data constructor for a @p VectorType.
class Vector : public Aggregate {
private:
    Vector(World& world, Defs args, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Base class for functional @p Insert and @p Extract.
class AggOp : public PrimOp {
protected:
    AggOp(NodeTag tag, const Type* type, Defs args, Debug dbg)
        : PrimOp(tag, type, args, dbg)
    {}

public:
    const Def* agg() const { return op(0); }
    const Def* index() const { return op(1); }

    friend class World;
};

/// Extracts from aggregate <tt>agg</tt> the element at position <tt>index</tt>.
class Extract : public AggOp {
private:
    Extract(const Def* agg, const Def* index, Debug dbg)
        : AggOp(Node_Extract, extracted_type(agg, index), {agg, index}, dbg)
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
    Insert(const Def* agg, const Def* index, const Def* value, Debug dbg)
        : AggOp(Node_Insert, agg->type(), {agg, index, value}, dbg)
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
    LEA(const Def* ptr, const Def* index, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const PtrType* ptr_type() const { return ptr()->type()->as<PtrType>(); } ///< Returns the PtrType from @p ptr().
    const Type* ptr_pointee() const { return ptr_type()->pointee(); }        ///< Returns the type referenced by @p ptr().

    friend class World;
};

/// Casts the underlying @p def to a dynamic value during @p partial_evaluation.
class Hlt : public PrimOp {
private:
    Hlt(const Def* def, Debug dbg)
        : PrimOp(Node_Hlt, def->type(), {def}, dbg)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/// Evaluates to @c true, if @p def is a literal.
class Known : public PrimOp {
private:
    Known(const Def* def, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/**
 * If a continuation typed def is wrapped in @p Run primop, it will be specialized into a callee whenever it is called.
 * Otherwise, this @p PrimOp evaluates to @p def.
 */
class Run : public PrimOp {
private:
    Run(const Def* def, Debug dbg)
        : PrimOp(Node_Run, def->type(), {def}, dbg)
    {}

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/**
 * A slot in a stack frame opend via @p Enter.
 * A @p Slot yields a pointer to the given <tt>type</tt>.
 * Loads from this address yield @p Bottom if the frame has already been closed.
 */
class Slot : public PrimOp {
private:
    Slot(const Type* type, const Def* frame, Debug dbg);

public:
    const Def* frame() const { return op(0); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Type* alloced_type() const { return type()->pointee(); }

private:
    virtual uint64_t vhash() const override;
    virtual bool equal(const PrimOp* other) const override;
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 */
class Global : public PrimOp {
private:
    Global(const Def* init, bool is_mutable, Debug dbg);

public:
    const Def* init() const { return op(0); }
    bool is_mutable() const { return is_mutable_; }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Type* alloced_type() const { return type()->pointee(); }
    virtual const char* op_name() const override;

    std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override { return murmur3(gid()); }
    virtual bool equal(const PrimOp* other) const override { return this == other; }
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    bool is_mutable_;

    friend class World;
};

/// Base class for all \p PrimOp%s taking and producing side-effects.
class MemOp : public PrimOp {
protected:
    MemOp(NodeTag tag, const Type* type, Defs args, Debug dbg)
        : PrimOp(tag, type, args, dbg)
    {
        assert(mem()->type()->isa<MemType>());
        assert(args.size() >= 1);
    }

public:
    const Def* mem() const { return op(0); }
    const Def* out_mem() const { return has_multiple_outs() ? out(0) : this; }

private:
    virtual uint64_t vhash() const override { return murmur3(gid()); }
    virtual bool equal(const PrimOp* other) const override { return this == other; }
};

/// Allocates memory on the heap.
class Alloc : public MemOp {
private:
    Alloc(const Type* type, const Def* mem, const Def* extra, Debug dbg);

public:
    const Def* extra() const { return op(1); }
    virtual bool has_multiple_outs() const override { return true; }
    const Def* out_ptr() const { return out(1); }
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    const PtrType* out_ptr_type() const { return type()->op(1)->as<PtrType>(); }
    const Type* alloced_type() const { return out_ptr_type()->pointee(); }
    static const Alloc* is_out_mem(const Def* def) { return is_out<0, Alloc>(def); }
    static const Alloc* is_out_ptr(const Def* def) { return is_out<1, Alloc>(def); }

private:
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Base class for @p Load and @p Store.
class Access : public MemOp {
protected:
    Access(NodeTag tag, const Type* type, Defs args, Debug dbg)
        : MemOp(tag, type, args, dbg)
    {
        assert(args.size() >= 2);
    }

public:
    const Def* ptr() const { return op(1); }
};

/// Loads with current effect <tt>mem</tt> from <tt>ptr</tt> to produce a pair of a new effect and the loaded value.
class Load : public Access {
private:
    Load(const Def* mem, const Def* ptr, Debug dbg);

public:
    virtual bool has_multiple_outs() const override { return true; }
    const Def* out_val() const { return out(1); }
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    const Type* out_val_type() const { return type()->op(1); }
    static const Load* is_out_mem(const Def* def) { return is_out<0, Load>(def); }
    static const Load* is_out_val(const Def* def) { return is_out<1, Load>(def); }

private:
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

    friend class World;
};

/// Stores with current effect <tt>mem</tt> <tt>value</tt> into <tt>ptr</tt> while producing a new effect.
class Store : public Access {
private:
    Store(const Def* mem, const Def* ptr, const Def* value, Debug dbg)
        : Access(Node_Store, mem->type(), {mem, ptr, value}, dbg)
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
    Enter(const Def* mem, Debug dbg);

    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;

public:
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    virtual bool has_multiple_outs() const override { return true; }
    const Def* out_frame() const { return out(1); }
    static const Enter* is_out_mem(const Def* def) { return is_out<0, Enter>(def); }
    static const Enter* is_out_frame(const Def* def) { return is_out<1, Enter>(def); }

    friend class World;
};

class Assembly : public MemOp {
public:
    enum Flags {
        NoFlag         = 0,
        HasSideEffects = 1 << 0,
        IsAlignStack   = 1 << 1,
        IsIntelDialect = 1 << 2,
    };

private:
    Assembly(const Type *type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, Debug dbg);

public:
    Defs inputs() const { return ops().skip_front(); }
    const Def* input(size_t i) const { return inputs()[i]; }
    size_t num_inputs() const { return inputs().size(); }
    virtual bool has_multiple_outs() const override { return true; }
    const std::string& asm_template() const { return asm_template_; }
    const ArrayRef<std::string> output_constraints() const { return output_constraints_; }
    const ArrayRef<std::string> input_constraints() const { return input_constraints_; }
    const ArrayRef<std::string> clobbers() const { return clobbers_; }
    bool has_sideeffects() const { return flags_ & HasSideEffects; }
    bool is_alignstack() const { return flags_ & IsAlignStack; }
    bool is_inteldialect() const { return flags_ & IsIntelDialect; }
    Flags flags() const { return flags_; }

private:
    virtual const Def* vrebuild(World& to, Defs ops, const Type* type) const override;
    virtual std::ostream& stream_assignment(std::ostream&) const override;

    std::string asm_template_;
    Array<std::string> output_constraints_, input_constraints_, clobbers_;
    Flags flags_;

    friend class World;
};

inline Assembly::Flags operator|(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) | static_cast<int>(rhs)); }
inline Assembly::Flags operator&(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) & static_cast<int>(rhs)); }
inline Assembly::Flags operator|=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs | rhs; }
inline Assembly::Flags operator&=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs & rhs; }

//------------------------------------------------------------------------------

template<int i, class T>
const T* PrimOp::is_out(const Def* def) {
    if (auto extract = def->isa<Extract>()) {
        if (is_primlit(extract->index(), i)) {
            if (auto res = extract->agg()->isa<T>())
                return res;
        }
    }
    return nullptr;
}

//------------------------------------------------------------------------------

template<class To>
using PrimOpMap     = GIDMap<const PrimOp*, To>;
using PrimOpSet     = GIDSet<const PrimOp*>;
using PrimOp2PrimOp = PrimOpMap<const PrimOp*>;

//------------------------------------------------------------------------------

bool is_from_match(const PrimOp*);

//------------------------------------------------------------------------------

}

#endif
