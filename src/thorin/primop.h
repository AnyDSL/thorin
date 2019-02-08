#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/enums.h"

namespace thorin {

//------------------------------------------------------------------------------

/// Base class for all @p PrimOp%s.
class PrimOp : public Def {
protected:
    PrimOp(NodeTag tag, const Def* type, Defs ops, Debug dbg)
        : Def(tag, type, ops, dbg)
    {}

public:
    const Def* out(size_t i) const;
    virtual bool has_multiple_outs() const { return false; }
    std::ostream& stream(std::ostream&) const override;

protected:

    /// Is @p def the @p i^th result of a @p T @p PrimOp?
    template<int i, class T> inline static const T* is_out(const Def* def);

    friend class World;
    friend class Cleaner;
    friend void Def::replace(Tracker) const;
};

//------------------------------------------------------------------------------

/// Akin to <tt>cond ? tval : fval</tt>.
class Select : public PrimOp {
private:
    Select(const Def* cond, const Def* tval, const Def* fval, Debug dbg)
        : PrimOp(Node_Select, tval->type(), {cond, tval, fval}, dbg)
    {
        assert(is_type_bool(cond->type()));
        assert(tval->type() == fval->type() && "types of both values must be equal");
        assert(!tval->type()->isa<Pi>() && "must not be a function");
    }

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

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

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const Def* of() const { return op(0)->type(); }

    friend class World;
};

/// Base class for all side-effect free binary \p PrimOp%s.
class BinOp : public PrimOp {
protected:
    BinOp(NodeTag tag, const Def* type, const Def* lhs, const Def* rhs, Debug dbg)
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
    {
        if ((tag == ArithOp_div || tag == ArithOp_rem) && is_type_i(type()->tag()))
            hash_ = murmur3(gid());
    }

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    ArithOpTag arithop_tag() const { return (ArithOpTag) tag(); }
    const char* op_name() const override;

    friend class World;
};

/// One of \p CmpTag compare.
class Cmp : public BinOp {
private:
    Cmp(CmpTag tag, const Def* lhs, const Def* rhs, Debug dbg);

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const PrimType* type() const { return BinOp::type()->as<PrimType>(); }
    CmpTag cmp_tag() const { return (CmpTag) tag(); }
    const char* op_name() const override;

    friend class World;
};

/// Base class for @p Bitcast and @p Cast.
class ConvOp : public PrimOp {
protected:
    ConvOp(NodeTag tag, const Def* from, const Def* to, Debug dbg)
        : PrimOp(tag, to, {from}, dbg)
    {}

public:
    const Def* from() const { return op(0); }
};

/// Converts <tt>from</tt> to type <tt>to</tt>.
class Cast : public ConvOp {
private:
    Cast(const Def* to, const Def* from, Debug dbg)
        : ConvOp(Node_Cast, from, to, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/// Reinterprets the bits of <tt>from</tt> as type <tt>to</tt>.
class Bitcast : public ConvOp {
private:
    Bitcast(const Def* to, const Def* from, Debug dbg)
        : ConvOp(Node_Bitcast, from, to, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/// Data constructor for a @p VariantType.
class Variant : public PrimOp {
private:
    Variant(const VariantType* variant_type, const Def* value, Debug dbg)
        : PrimOp(Node_Variant, variant_type, {value}, dbg)
    {
        assert(std::find(variant_type->ops().begin(), variant_type->ops().end(), value->type()) != variant_type->ops().end());
    }

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const VariantType* type() const { return PrimOp::type()->as<VariantType>(); }

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
    LEA(const Def* type, const Def* ptr, const Def* index, Debug dbg)
        : PrimOp(Node_LEA, type, {ptr, index}, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const PtrType* ptr_type() const { return ptr()->type()->as<PtrType>(); } ///< Returns the PtrType from @p ptr().
    const Def* ptr_pointee() const { return ptr_type()->pointee(); }        ///< Returns the type referenced by @p ptr().

    friend class World;
};

/// Casts the underlying @p def to a dynamic value during @p partial_evaluation.
class Hlt : public PrimOp {
private:
    Hlt(const Def* def, Debug dbg)
        : PrimOp(Node_Hlt, def->type(), {def}, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/// Evaluates to @c true, if @p def is a literal.
class Known : public PrimOp {
private:
    Known(const Def* def, Debug dbg);

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const Def* def() const { return op(0); }

    friend class World;
};

/**
 * If a lam typed def is wrapped in @p Run primop, it will be specialized into a callee whenever it is called.
 * Otherwise, this @p PrimOp evaluates to @p def.
 */
class Run : public PrimOp {
private:
    Run(const Def* def, Debug dbg)
        : PrimOp(Node_Run, def->type(), {def}, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

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
    Slot(const Def* type, const Def* frame, Debug dbg)
        : PrimOp(Node_Slot, type, {frame}, dbg)
    {
        hash_ = murmur3(gid()); // HACK
        assert(frame->type()->isa<FrameType>());
    }

public:
    const Def* frame() const { return op(0); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Def* alloced_type() const { return type()->pointee(); }

private:
    bool equal(const Def* other) const override;
    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 */
class Global : public PrimOp {
private:
    struct Extra { bool is_mutable_; }; // TODO remove

    Global(const Def* type, const Def* init, bool is_mutable, Debug dbg)
        : PrimOp(Node_Global, type, {init}, dbg)
    {
        extra<Extra>().is_mutable_ = is_mutable;
        hash_ = murmur3(gid()); // HACK
        assert(is_const(init));
    }

public:
    const Def* init() const { return op(0); }
    bool is_mutable() const { return extra<Extra>().is_mutable_; }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Def* alloced_type() const { return type()->pointee(); }
    const char* op_name() const override;

    std::ostream& stream(std::ostream&) const override;

private:
    bool equal(const Def* other) const override { return this == other; }
    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/// Base class for all \p PrimOp%s taking and producing side-effects.
class MemOp : public PrimOp {
protected:
    MemOp(NodeTag tag, const Def* type, Defs args, Debug dbg)
        : PrimOp(tag, type, args, dbg)
    {
        hash_ = murmur3(gid()); // HACK
        assert(mem()->type()->isa<MemType>());
        assert(args.size() >= 1);
    }

public:
    const Def* mem() const { return op(0); }
    const Def* out_mem() const { return has_multiple_outs() ? out(0) : this; }

private:
    bool equal(const Def* other) const override { return this == other; }
};

/// Allocates memory on the heap.
class Alloc : public MemOp {
private:
    Alloc(const Def* type, const Def* mem, Debug dbg)
        : MemOp(Node_Alloc, type, {mem}, dbg)
    {}

public:
    bool has_multiple_outs() const override { return true; }
    const Def* out_ptr() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const PtrType* out_ptr_type() const { return type()->op(1)->as<PtrType>(); }
    const Def* alloced_type() const { return out_ptr_type()->pointee(); }
    static const Alloc* is_out_mem(const Def* def) { return is_out<0, Alloc>(def); }
    static const Alloc* is_out_ptr(const Def* def) { return is_out<1, Alloc>(def); }

private:
    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/// Base class for @p Load and @p Store.
class Access : public MemOp {
protected:
    Access(NodeTag tag, const Def* type, Defs args, Debug dbg)
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
    Load(const Def* type, const Def* mem, const Def* ptr, Debug dbg)
        : Access(Node_Load, type, {mem, ptr}, dbg)
    {}

public:
    bool has_multiple_outs() const override { return true; }
    const Def* out_val() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const Def* out_val_type() const { return type()->op(1); }
    static const Load* is_out_mem(const Def* def) { return is_out<0, Load>(def); }
    static const Load* is_out_val(const Def* def) { return is_out<1, Load>(def); }

private:
    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/// Stores with current effect <tt>mem</tt> <tt>value</tt> into <tt>ptr</tt> while producing a new effect.
class Store : public Access {
private:
    Store(const Def* mem, const Def* ptr, const Def* value, Debug dbg)
        : Access(Node_Store, mem->type(), {mem, ptr, value}, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const Def* val() const { return op(2); }
    const MemType* type() const { return Access::type()->as<MemType>(); }

    friend class World;
};

/// Creates a stack \p Frame with current effect <tt>mem</tt>.
class Enter : public MemOp {
private:
    Enter(const Def* type, const Def* mem, Debug dbg)
        : MemOp(Node_Enter, type, {mem}, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    bool has_multiple_outs() const override { return true; }
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

    struct Extra {
        std::string asm_template_;
        Array<std::string> output_constraints_, input_constraints_, clobbers_;
        Flags flags_;
    };

private:
    Assembly(const Def *type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, Debug dbg);
    ~Assembly() override;

public:
    Defs inputs() const { return ops().skip_front(); }
    const Def* input(size_t i) const { return inputs()[i]; }
    size_t num_inputs() const { return inputs().size(); }
    bool has_multiple_outs() const override { return true; }
    const std::string& asm_template() const { return extra<Extra>().asm_template_; }
    const ArrayRef<std::string> output_constraints() const { return extra<Extra>().output_constraints_; }
    const ArrayRef<std::string> input_constraints() const { return extra<Extra>().input_constraints_; }
    const ArrayRef<std::string> clobbers() const { return extra<Extra>().clobbers_; }
    bool has_sideeffects() const { return flags() & HasSideEffects; }
    bool is_alignstack() const { return flags() & IsAlignStack; }
    bool is_inteldialect() const { return flags() & IsIntelDialect; }
    Flags flags() const { return extra<Extra>().flags_; }
    const Def* rebuild(World& to, const Def* type, Defs ops) const override;
    std::ostream& stream_assignment(std::ostream&) const override;

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

}

#endif
