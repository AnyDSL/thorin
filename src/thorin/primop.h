#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/config.h"
#include "thorin/util.h"
#include "thorin/enums.h"

namespace thorin {

//------------------------------------------------------------------------------

/// Base class for all @p PrimOp%s.
class PrimOp : public Def {
protected:
    PrimOp(NodeTag tag, RebuildFn rebuild, const Def* type, Defs ops, const Def* dbg)
        : Def(tag, rebuild, type, ops, dbg)
    {}

public:
    std::ostream& stream(std::ostream&) const override;

    friend class World;
    friend class Cleaner;
    friend void Def::replace(Tracker) const;
};

//------------------------------------------------------------------------------

/// Akin to <tt>cond ? tval : fval</tt>.
class Select : public PrimOp {
private:
    Select(const Def* cond, const Def* tval, const Def* fval, const Def* dbg)
        : PrimOp(Node_Select, rebuild, tval->type(), {cond, tval, fval}, dbg)
    {
        assert(is_type_bool(cond->type()));
        assert(tval->type() == fval->type() && "types of both values must be equal");
    }

public:
    const Def* cond() const { return op(0); }
    const Def* tval() const { return op(1); }
    const Def* fval() const { return op(2); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Get number of bytes needed for any value (including bottom) of a given @p Type.
class SizeOf : public PrimOp {
private:
    SizeOf(const Def* def, const Def* dbg);

public:
    const Def* of() const { return op(0)->type(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Base class for all side-effect free binary \p PrimOp%s.
class BinOp : public PrimOp {
protected:
    BinOp(NodeTag tag, RebuildFn rebuild, const Def* type, const Def* lhs, const Def* rhs, const Def* dbg)
        : PrimOp(tag, rebuild, type, {lhs, rhs}, dbg)
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
    ArithOp(ArithOpTag tag, const Def* lhs, const Def* rhs, const Def* dbg)
        : BinOp((NodeTag) tag, rebuild, lhs->type(), lhs, rhs, dbg)
    {
        // TODO remove this and make div/rem proper nodes *with* side-effects
        if ((tag == ArithOp_div || tag == ArithOp_rem) && is_type_i(type()->tag()))
            hash_ = murmur3(gid());
    }

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    ArithOpTag arithop_tag() const { return (ArithOpTag) tag(); }
    const char* op_name() const override;
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// One of \p CmpTag compare.
class Cmp : public BinOp {
private:
    Cmp(CmpTag tag, const Def* lhs, const Def* rhs, const Def* dbg);

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    CmpTag cmp_tag() const { return (CmpTag) tag(); }
    const char* op_name() const override;
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Base class for @p Bitcast and @p Cast.
class ConvOp : public PrimOp {
protected:
    ConvOp(NodeTag tag, RebuildFn rebuild, const Def* from, const Def* to, const Def* dbg)
        : PrimOp(tag, rebuild, to, {from}, dbg)
    {}

public:
    const Def* from() const { return op(0); }
};

/// Converts <tt>from</tt> to type <tt>to</tt>.
class Cast : public ConvOp {
private:
    Cast(const Def* to, const Def* from, const Def* dbg)
        : ConvOp(Node_Cast, rebuild, from, to, dbg)
    {}

public:
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Reinterprets the bits of <tt>from</tt> as type <tt>to</tt>.
class Bitcast : public ConvOp {
private:
    Bitcast(const Def* to, const Def* from, const Def* dbg)
        : ConvOp(Node_Bitcast, rebuild, from, to, dbg)
    {}

public:
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Data constructor for a @p VariantType.
class Variant : public PrimOp {
private:
    Variant(const VariantType* variant_type, const Def* value, const Def* dbg)
        : PrimOp(Node_Variant, rebuild, variant_type, {value}, dbg)
    {
        assert(std::find(variant_type->ops().begin(), variant_type->ops().end(), value->type()) != variant_type->ops().end());
    }

public:
    const VariantType* type() const { return PrimOp::type()->as<VariantType>(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

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
    LEA(const Def* type, const Def* ptr, const Def* index, const Def* dbg)
        : PrimOp(Node_LEA, rebuild, type, {ptr, index}, dbg)
    {}

public:
    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const PtrType* ptr_type() const { return ptr()->type()->as<PtrType>(); } ///< Returns the PtrType from @p ptr().
    const Def* ptr_pointee() const { return ptr_type()->pointee(); }        ///< Returns the type referenced by @p ptr().
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Casts the underlying @p def to a dynamic value during @p partial_evaluation.
class Hlt : public PrimOp {
private:
    Hlt(const Def* def, const Def* dbg)
        : PrimOp(Node_Hlt, rebuild, def->type(), {def}, dbg)
    {}

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Evaluates to @c true, if @p def is a literal.
class Known : public PrimOp {
private:
    Known(const Def* def, const Def* dbg);

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/**
 * If a lam typed def is wrapped in @p Run primop, it will be specialized into a callee whenever it is called.
 * Otherwise, this @p PrimOp evaluates to @p def.
 */
class Run : public PrimOp {
private:
    Run(const Def* def, const Def* dbg)
        : PrimOp(Node_Run, rebuild, def->type(), {def}, dbg)
    {}

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 */
class Global : public PrimOp {
private:
    struct Extra { bool is_mutable_; }; // TODO remove

    Global(const Def* type, const Def* init, bool is_mutable, const Def* dbg)
        : PrimOp(Node_Global, rebuild, type, {init}, dbg)
    {
        extra<Extra>().is_mutable_ = is_mutable;
        hash_ = murmur3(gid()); // HACK
    }

public:
    const Def* init() const { return op(0); }
    bool is_mutable() const { return extra<Extra>().is_mutable_; }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Def* alloced_type() const { return type()->pointee(); }
    const char* op_name() const override;

    bool equal(const Def* other) const override { return this == other; }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

/// Base class for all \p PrimOp%s taking and producing side-effects.
class MemOp : public PrimOp {
protected:
    MemOp(NodeTag tag, RebuildFn rebuild, const Def* type, Defs args, const Def* dbg)
        : PrimOp(tag, rebuild, type, args, dbg)
    {
        assert(mem()->type()->isa<MemType>());
        assert(args.size() >= 1);
    }

public:
    const Def* mem() const { return op(0); }
    const Def* out_mem() const { return out(0); }
};

/// Allocates memory on the heap.
class Alloc : public MemOp {
private:
    Alloc(const Def* type, const Def* mem, const Def* dbg)
        : MemOp(Node_Alloc, rebuild, type, {mem}, dbg)
    {}

public:
    const Def* out_ptr() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const PtrType* out_ptr_type() const { return type()->op(1)->as<PtrType>(); }
    const Def* alloced_type() const { return out_ptr_type()->pointee(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Allocates memory on the stack.
/// TODO eventually substitute with Alloc
class Slot : public MemOp {
private:
    Slot(const Def* type, const Def* mem, const Def* dbg)
        : MemOp(Node_Slot, rebuild, type, {mem}, dbg)
    {}

public:
    const Def* out_ptr() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const PtrType* out_ptr_type() const { return type()->op(1)->as<PtrType>(); }
    const Def* alloced_type() const { return out_ptr_type()->pointee(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Base class for @p Load and @p Store.
class Access : public MemOp {
protected:
    Access(NodeTag tag, RebuildFn rebuild, const Def* type, Defs args, const Def* dbg)
        : MemOp(tag, rebuild, type, args, dbg)
    {
        assert(args.size() >= 2);
    }

public:
    const Def* ptr() const { return op(1); }
};

/// Loads with current effect <tt>mem</tt> from <tt>ptr</tt> to produce a pair of a new effect and the loaded value.
class Load : public Access {
private:
    Load(const Def* type, const Def* mem, const Def* ptr, const Def* dbg)
        : Access(Node_Load, rebuild, type, {mem, ptr}, dbg)
    {}

public:
    const Def* out_val() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const Def* out_val_type() const { return type()->op(1); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    friend class World;
};

/// Stores with current effect <tt>mem</tt> <tt>value</tt> into <tt>ptr</tt> while producing a new effect.
class Store : public Access {
private:
    Store(const Def* mem, const Def* ptr, const Def* value, const Def* dbg)
        : Access(Node_Store, rebuild, mem->type(), {mem, ptr, value}, dbg)
    {}

public:
    const Def* val() const { return op(2); }
    const MemType* type() const { return Access::type()->as<MemType>(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

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

    struct Extra {
        std::string asm_template_;
        Array<std::string> output_constraints_, input_constraints_, clobbers_;
        Flags flags_;
    };

private:
    Assembly(const Def *type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, const Def* dbg);
    ~Assembly() override;

public:
    Defs inputs() const { return ops().skip_front(); }
    const Def* input(size_t i) const { return inputs()[i]; }
    size_t num_inputs() const { return inputs().size(); }
    const std::string& asm_template() const { return extra<Extra>().asm_template_; }
    const ArrayRef<std::string> output_constraints() const { return extra<Extra>().output_constraints_; }
    const ArrayRef<std::string> input_constraints() const { return extra<Extra>().input_constraints_; }
    const ArrayRef<std::string> clobbers() const { return extra<Extra>().clobbers_; }
    bool has_sideeffects() const { return flags() & HasSideEffects; }
    bool is_alignstack() const { return flags() & IsAlignStack; }
    bool is_inteldialect() const { return flags() & IsIntelDialect; }
    Flags flags() const { return extra<Extra>().flags_; }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);
    std::ostream& stream_assignment(std::ostream&) const override;

    friend class World;
};

inline Assembly::Flags operator|(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) | static_cast<int>(rhs)); }
inline Assembly::Flags operator&(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) & static_cast<int>(rhs)); }
inline Assembly::Flags operator|=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs | rhs; }
inline Assembly::Flags operator&=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs & rhs; }

template<class To>
using PrimOpMap     = GIDMap<const PrimOp*, To>;
using PrimOpSet     = GIDSet<const PrimOp*>;
using PrimOp2PrimOp = PrimOpMap<const PrimOp*>;

}

#endif
