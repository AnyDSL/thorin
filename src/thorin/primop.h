#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/type.h"
#include "thorin/enums.h"
#include "thorin/util/hash.h"

namespace thorin {

class Literal : public Def {
protected:
    Literal(World& world, NodeTag tag, const Type* type, Debug dbg)
        : Def(world, tag, type, Defs{}, dbg)
    {}
};

/// This literal represents 'no value'.
class Bottom : public Literal {
private:
    Bottom(World& world, const Type* type, Debug dbg)
        : Literal(world, Node_Bottom, type, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// This literal represents 'any value'.
class Top : public Literal {
private:
    Top(World& world, const Type* type, Debug dbg)
        : Literal(world, Node_Top, type, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

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

private:
    hash_t vhash() const override;
    bool equal(const Def*) const override;
    const Def* rebuild(World&, const Type*, Defs) const override;

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
class Select : public Def {
private:
    Select(World& world, const Def* cond, const Def* tval, const Def* fval, Debug dbg)
        : Def(world, Node_Select, tval->type(), {cond, tval, fval}, dbg)
    {
        assert(is_type_bool(cond->type()));
        assert(tval->type() == fval->type() && "types of both values must be equal");
        assert(!tval->type()->isa<FnType>() && "must not be a function");
    }

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Def* cond() const { return op(0); }
    const Def* tval() const { return op(1); }
    const Def* fval() const { return op(2); }

    friend class World;
};

/// Get the alignment in number of bytes needed for any value (including bottom) of a given @p Type.
class AlignOf : public Def {
private:
    AlignOf(World& world, const Def* def, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Type* of() const { return op(0)->type(); }

    friend class World;
};

/// Get number of bytes needed for any value (including bottom) of a given @p Type.
class SizeOf : public Def {
private:
    SizeOf(World& world, const Def* def, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Type* of() const { return op(0)->type(); }

    friend class World;
};

/// Base class for all side-effect free binary \p Def%s.
class BinOp : public Def {
protected:
    BinOp(World& world, NodeTag tag, const Type* type, const Def* lhs, const Def* rhs, Debug dbg)
        : Def(world, tag, type, {lhs, rhs}, dbg)
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
    ArithOp(ArithOpTag tag, World& world, const Def* lhs, const Def* rhs, Debug dbg)
        : BinOp(world, (NodeTag) tag, lhs->type(), lhs, rhs, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    ArithOpTag arithop_tag() const { return (ArithOpTag) tag(); }

    friend class World;
};

/// One of \p CmpTag compare.
class Cmp : public BinOp {
private:
    Cmp(CmpTag tag, World& world, const Def* lhs, const Def* rhs, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    CmpTag cmp_tag() const { return (CmpTag) tag(); }

    friend class World;
};

/// Common mathematical function such as `sin()` or `cos()`.
class MathOp : public Def {
private:
    MathOp(World& world, MathOpTag tag, const Type* type, Defs args, Debug dbg)
        : Def(world, (NodeTag)tag, type, args, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const PrimType* type() const { return Def::type()->as<PrimType>(); }
    MathOpTag mathop_tag() const { return (MathOpTag) tag(); }

    friend class World;
};

/// Base class for @p Bitcast and @p Cast.
class ConvOp : public Def {
protected:
    ConvOp(World& world, NodeTag tag, const Def* from, const Type* to, Debug dbg)
        : Def(world, tag, to, {from}, dbg)
    {}

public:
    const Def* from() const { return op(0); }
};

/// Converts <tt>from</tt> to type <tt>to</tt>.
class Cast : public ConvOp {
private:
    Cast(World& world, const Type* to, const Def* from, Debug dbg)
        : ConvOp(world, Node_Cast, from, to, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// Reinterprets the bits of <tt>from</tt> as type <tt>to</tt>.
class Bitcast : public ConvOp {
private:
    Bitcast(World& world, const Type* to, const Def* from, Debug dbg)
        : ConvOp(world, Node_Bitcast, from, to, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// Base class for all aggregate data constructers.
class Aggregate : public Def {
protected:
    Aggregate(World& world, NodeTag tag, Defs args, Debug dbg)
        : Def(world, tag, nullptr /*set later*/, args, dbg)
    {}
};

/// Data constructor for a \p DefiniteArrayType.
class DefiniteArray : public Aggregate {
private:
    DefiniteArray(World& world, const Type* elem, Defs args, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

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

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const IndefiniteArrayType* type() const { return Aggregate::type()->as<IndefiniteArrayType>(); }
    const Type* elem_type() const { return type()->elem_type(); }

    friend class World;
};

/// Data constructor for a @p TupleType.
class Tuple : public Aggregate {
private:
    Tuple(World& world, Defs args, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const TupleType* type() const { return Aggregate::type()->as<TupleType>(); }

    friend class World;
};

/// Data constructor for a @p VariantType.
class Variant : public Def {
private:
    Variant(World& world, const VariantType* variant_type, const Def* value, size_t index, Debug dbg)
        : Def(world, Node_Variant, variant_type, {value}, dbg), index_(index)
    {
        assert(variant_type->op(index) == value->type());
    }

    const Def* rebuild(World&, const Type*, Defs) const override;
    hash_t vhash() const override;
    bool equal(const Def*) const override;

    size_t index_;

public:
    const VariantType* type() const { return Def::type()->as<VariantType>(); }
    size_t index() const { return index_; }
    const Def* value() const { return op(0); }

    friend class World;
};

/// Yields the tag/index for this variant in the supplied integer type
class VariantIndex : public Def {
private:
    VariantIndex(World& world, const Type* int_type, const Def* value, Debug dbg)
        : Def(world, Node_VariantIndex, int_type, {value}, dbg)
    {
        assert(value->type()->isa<VariantType>());
        assert(is_type_s(int_type) || is_type_u(int_type));
    }

    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

class VariantExtract : public Def {
private:
    VariantExtract(World& world, const Type* type, const Def* value, size_t index, Debug dbg)
        : Def(world, Node_VariantExtract, type, {value}, dbg), index_(index)
    {
        assert(value->type()->as<VariantType>()->op(index) == type);
    }

    const Def* rebuild(World&, const Type*, Defs) const override;
    hash_t vhash() const override;
    bool equal(const Def*) const override;

    size_t index_;

public:
    size_t index() const { return index_; }
    const Def* value() const { return op(0); }

    friend class World;
};

/// Data constructor for a @p ClosureType.
class Closure : public Aggregate {
private:
    Closure(World& world, const ClosureType* closure_type, const Def* fn, const Def* env, Debug dbg)
        : Aggregate(world, Node_Closure, {fn, env}, dbg)
    {
        set_type(closure_type);
    }

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    static const Type*    environment_type(World&);
    static const PtrType* environment_ptr_type(World&);

    Continuation* fn() const;

    friend class World;
};

/// Data constructor for a @p StructType.
class StructAgg : public Aggregate {
private:
    StructAgg(World& world, const StructType* struct_type, Defs args, Debug dbg)
        : Aggregate(world, Node_StructAgg, args, dbg)
    {
#if THORIN_ENABLE_CHECKS
        assert(struct_type->num_ops() == args.size());
        for (size_t i = 0, e = args.size(); i != e; ++i)
            assert(struct_type->op(i) == args[i]->type());
#endif
        set_type(struct_type);
    }

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const StructType* type() const { return Aggregate::type()->as<StructType>(); }

    friend class World;
};

/// Data constructor for a @p VectorType.
class Vector : public Aggregate {
private:
    Vector(World& world, Defs args, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// Base class for functional @p Insert and @p Extract.
class AggOp : public Def {
protected:
    AggOp(World& world, NodeTag tag, const Type* type, Defs args, Debug dbg)
        : Def(world, tag, type, args, dbg)
    {}

public:
    const Def* agg() const { return op(0); }
    const Def* index() const { return op(1); }

    friend class World;
};

/// Extracts from aggregate <tt>agg</tt> the element at position <tt>index</tt>.
class Extract : public AggOp {
private:
    Extract(World& world, const Def* agg, const Def* index, Debug dbg)
        : AggOp(world, Node_Extract, extracted_type(agg, index), {agg, index}, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

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
    Insert(World& world, const Def* agg, const Def* index, const Def* value, Debug dbg)
        : AggOp(world, Node_Insert, agg->type(), {agg, index, value}, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

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
class LEA : public Def {
private:
    LEA(World& world, const Def* ptr, const Def* index, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }
    const PtrType* type() const { return Def::type()->as<PtrType>(); }
    const PtrType* ptr_type() const { return ptr()->type()->as<PtrType>(); } ///< Returns the PtrType from @p ptr().
    const Type* ptr_pointee() const { return ptr_type()->pointee(); }        ///< Returns the type referenced by @p ptr().

    friend class World;
};

/// Casts the underlying @p def to a dynamic value during @p partial_evaluation.
class Hlt : public Def {
private:
    Hlt(World& world, const Def* def, Debug dbg)
        : Def(world, Node_Hlt, def->type(), {def}, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/// Evaluates to @c true, if @p def is a literal.
class Known : public Def {
private:
    Known(World& world, const Def* def, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/**
 * If a continuation typed def is wrapped in @p Run primop, it will be specialized into a callee whenever it is called.
 * Otherwise, this @p Def evaluates to @p def.
 */
class Run : public Def {
private:
    Run(World& world, const Def* def, Debug dbg)
        : Def(world, Node_Run, def->type(), {def}, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/**
 * A slot in a stack frame opend via @p Enter.
 * A @p Slot yields a pointer to the given <tt>type</tt>.
 * Loads from this address yield @p Bottom if the frame has already been closed.
 */
class Slot : public Def {
private:
    Slot(World& world, const Type* type, const Def* frame, Debug dbg);

public:
    const Def* frame() const { return op(0); }
    const PtrType* type() const { return Def::type()->as<PtrType>(); }
    const Type* alloced_type() const { return type()->pointee(); }

private:
    hash_t vhash() const override;
    bool equal(const Def*) const override;
    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 */
class Global : public Def {
private:
    Global(World& world, const Def* init, bool is_mutable, Debug dbg);

public:
    const Def* init() const { return op(0); }
    bool is_mutable() const { return is_mutable_; }
    const PtrType* type() const { return Def::type()->as<PtrType>(); }
    const Type* alloced_type() const { return type()->pointee(); }
    const char* op_name() const override;

    bool is_external() const;
    void set_init(const Def* new_init) { unset_op(0); set_op(0, new_init); }
    const Def* rebuild(World&, const Type*, Defs) const override;

private:
    hash_t vhash() const override { return murmur3(gid()); }
    bool equal(const Def* other) const override { return this == other; }

    bool is_mutable_;

    friend class World;
};

/// Base class for all \p Def%s taking and producing side-effects.
class MemOp : public Def {
protected:
    MemOp(World& world, NodeTag tag, const Type* type, Defs args, Debug dbg)
        : Def(world, tag, type, args, dbg)
    {
        assert(mem()->type()->isa<MemType>());
        assert(args.size() >= 1);
    }

public:
    const Def* mem() const { return op(0); }
    const Def* out_mem() const { return has_multiple_outs() ? out(0) : this; }

private:
    hash_t vhash() const override { return murmur3(gid()); }
    bool equal(const Def* other) const override { return this == other; }
};

/// Allocates memory on the heap.
class Alloc : public MemOp {
private:
    Alloc(World& world, const Type* type, const Def* mem, const Def* extra, Debug dbg);

public:
    const Def* extra() const { return op(1); }
    bool has_multiple_outs() const override { return true; }
    const Def* out_ptr() const { return out(1); }
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    const PtrType* out_ptr_type() const { return type()->op(1)->as<PtrType>(); }
    const Type* alloced_type() const { return out_ptr_type()->pointee(); }

private:
    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

class Release : public MemOp {
private:
    Release(World& world, const Def* mem, const Def* alloc, Debug dbg);

public:
    const Def* alloc() const { return op(1); }

private:
    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};


/// Base class for @p Load and @p Store.
class Access : public MemOp {
protected:
    Access(World& world, NodeTag tag, const Type* type, Defs args, Debug dbg)
        : MemOp(world, tag, type, args, dbg)
    {
        assert(args.size() >= 2);
    }

public:
    const Def* ptr() const { return op(1); }
};

/// Loads with current effect <tt>mem</tt> from <tt>ptr</tt> to produce a pair of a new effect and the loaded value.
class Load : public Access {
private:
    Load(World& world, const Def* mem, const Def* ptr, Debug dbg);

public:
    bool has_multiple_outs() const override { return true; }
    const Def* out_val() const { return out(1); }
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    const Type* out_val_type() const { return type()->op(1)->as<Type>(); }

private:
    const Def* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// Stores with current effect <tt>mem</tt> <tt>value</tt> into <tt>ptr</tt> while producing a new effect.
class Store : public Access {
private:
    Store(World& world, const Def* mem, const Def* ptr, const Def* value, Debug dbg)
        : Access(world, Node_Store, mem->type(), {mem, ptr, value}, dbg)
    {}

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const Def* val() const { return op(2); }
    const MemType* type() const { return Access::type()->as<MemType>(); }

    friend class World;
};

/// Creates a stack \p Frame with current effect <tt>mem</tt>.
class Enter : public MemOp {
private:
    Enter(World& world, const Def* mem, Debug dbg);

    const Def* rebuild(World&, const Type*, Defs) const override;

public:
    const TupleType* type() const { return MemOp::type()->as<TupleType>(); }
    bool has_multiple_outs() const override { return true; }
    const Def* out_frame() const { return out(1); }

    static const Enter* is_out_mem(const Def*);

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
    Assembly(World& world, const Type *type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, Debug dbg);

public:
    Defs inputs() const { return ops().skip_front(); }
    const Def* input(size_t i) const { return inputs()[i]; }
    size_t num_inputs() const { return inputs().size(); }
    bool has_multiple_outs() const override { return true; }
    const std::string& asm_template() const { return asm_template_; }
    const ArrayRef<std::string> output_constraints() const { return output_constraints_; }
    const ArrayRef<std::string> input_constraints() const { return input_constraints_; }
    const ArrayRef<std::string> clobbers() const { return clobbers_; }
    bool has_sideeffects() const { return flags_ & HasSideEffects; }
    bool is_alignstack() const { return flags_ & IsAlignStack; }
    bool is_inteldialect() const { return flags_ & IsIntelDialect; }
    Flags flags() const { return flags_; }

private:
    const Def* rebuild(World&, const Type*, Defs) const override;

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
const T* Def::is_out(const Def* def) {
    if (auto extract = def->isa<Extract>()) {
        if (is_primlit(extract->index(), i)) {
            if (auto res = extract->agg()->isa<T>())
                return res;
        }
    }
    return nullptr;
}

//------------------------------------------------------------------------------

}

#endif
