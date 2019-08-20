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
    PrimOp(NodeTag tag, RebuildFn rebuild, const Def* type, Defs ops, uint64_t flags, const Def* dbg)
        : Def(tag, rebuild, type, ops, flags, dbg)
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
        : PrimOp(Node_Select, rebuild, tval->type(), {cond, tval, fval}, 0, dbg)
    {
        assert(is_type_bool(cond->type()));
        assert(tval->type() == fval->type() && "types of both values must be equal");
    }

public:
    const Def* cond() const { return op(0); }
    const Def* tval() const { return op(1); }
    const Def* fval() const { return op(2); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Select;
    friend class World;
};

/// Get number of bytes needed for any value (including bottom) of a given @p Type.
class SizeOf : public PrimOp {
private:
    SizeOf(const Def* def, const Def* dbg);

public:
    const Def* of() const { return op(0)->type(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::SizeOf;
    friend class World;
};

/// One of \p ArithOpTag arithmetic operation.
class ArithOp : public PrimOp {
private:
    ArithOp(ArithOpTag tag, const Def* lhs, const Def* rhs, const Def* dbg)
        : PrimOp(Node_ArithOp, rebuild, lhs->type(), {lhs, rhs}, tag, dbg)
    {
        assert(lhs->type() == rhs->type() && "types are not equal");
        // TODO remove this and make div/rem proper nodes *with* side-effects
        if ((tag == ArithOp_div || tag == ArithOp_rem) && is_type_i(type()))
            hash_ = murmur3(gid());
    }

public:
    const Def* lhs() const { return op(0); }
    const Def* rhs() const { return op(1); }
    const PrimType* type() const { return PrimOp::type()->as<PrimType>(); }
    ArithOpTag arithop_tag() const { return (ArithOpTag) flags(); }
    const char* op_name() const override;
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::ArithOp;
    friend class World;
};

/// One of \p CmpTag compare.
class Cmp : public PrimOp {
private:
    Cmp(CmpTag tag, const Def* lhs, const Def* rhs, const Def* dbg);

public:
    const Def* lhs() const { return op(0); }
    const Def* rhs() const { return op(1); }
    const PrimType* type() const { return PrimOp::type()->as<PrimType>(); }
    CmpTag cmp_tag() const { return (CmpTag) flags(); }
    const char* op_name() const override;
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Cmp;
    friend class World;
};

/// Converts <tt>from</tt> to type <tt>to</tt>.
class Cast : public PrimOp {
private:
    Cast(const Def* to, const Def* from, const Def* dbg)
        : PrimOp(Node_Cast, rebuild, to, {from}, 0, dbg)
    {}

public:
    const Def* from() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Cast;
    friend class World;
};

/// Reinterprets the bits of <tt>from</tt> as type <tt>to</tt>.
class Bitcast : public PrimOp {
private:
    Bitcast(const Def* to, const Def* from, const Def* dbg)
        : PrimOp(Node_Bitcast, rebuild, to, {from}, 0, dbg)
    {}

public:
    const Def* from() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Bitcast;
    friend class World;
};

/// Data constructor for a @p VariantType.
class Variant : public PrimOp {
private:
    Variant(const VariantType* variant_type, const Def* value, const Def* dbg)
        : PrimOp(Node_Variant, rebuild, variant_type, {value}, 0, dbg)
    {
        assert(std::find(variant_type->ops().begin(), variant_type->ops().end(), value->type()) != variant_type->ops().end());
    }

public:
    const VariantType* type() const { return PrimOp::type()->as<VariantType>(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Variant;
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
        : PrimOp(Node_LEA, rebuild, type, {ptr, index}, 0, dbg)
    {}

public:
    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const PtrType* ptr_type() const { return ptr()->type()->as<PtrType>(); } ///< Returns the PtrType from @p ptr().
    const Def* ptr_pointee() const { return ptr_type()->pointee(); }        ///< Returns the type referenced by @p ptr().
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::LEA;
    friend class World;
};

/// Casts the underlying @p def to a dynamic value during @p partial_evaluation.
class Hlt : public PrimOp {
private:
    Hlt(const Def* def, const Def* dbg)
        : PrimOp(Node_Hlt, rebuild, def->type(), {def}, 0, dbg)
    {}

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Hlt;
    friend class World;
};

/// Evaluates to @c true, if @p def is a literal.
class Known : public PrimOp {
private:
    Known(const Def* def, const Def* dbg);

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Known;
    friend class World;
};

/**
 * If a lam typed def is wrapped in @p Run primop, it will be specialized into a callee whenever it is called.
 * Otherwise, this @p PrimOp evaluates to @p def.
 */
class Run : public PrimOp {
private:
    Run(const Def* def, const Def* dbg)
        : PrimOp(Node_Run, rebuild, def->type(), {def}, 0, dbg)
    {}

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Run;
    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 */
class Global : public PrimOp {
private:
    Global(const Def* type, const Def* id, const Def* init, bool is_mutable, const Def* dbg)
        : PrimOp(Node_Global, rebuild, type, {id, init}, is_mutable, dbg)
    {}

public:
    /// This thing's sole purpose is to differentiate on global from another.
    const Def* id() const { return op(0); }
    const Def* init() const { return op(1); }
    bool is_mutable() const { return flags(); }
    const PtrType* type() const { return PrimOp::type()->as<PtrType>(); }
    const Def* alloced_type() const { return type()->pointee(); }
    const char* op_name() const override;

    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Tag = Tags::Global;
    friend class World;
};

/// Base class for all \p PrimOp%s taking and producing side-effects.
class MemOp : public PrimOp {
protected:
    MemOp(NodeTag tag, RebuildFn rebuild, const Def* type, Defs args, uint64_t flags, const Def* dbg)
        : PrimOp(tag, rebuild, type, args, flags, dbg)
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
        : MemOp(Node_Alloc, rebuild, type, {mem}, 0, dbg)
    {}

public:
    const Def* out_ptr() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const PtrType* out_ptr_type() const { return type()->op(1)->as<PtrType>(); }
    const Def* alloced_type() const { return out_ptr_type()->pointee(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Alloc;
    friend class World;
};

/// Allocates memory on the stack.
/// TODO eventually substitute with Alloc
class Slot : public MemOp {
private:
    Slot(const Def* type, const Def* mem, const Def* dbg)
        : MemOp(Node_Slot, rebuild, type, {mem}, 0, dbg)
    {}

public:
    const Def* out_ptr() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const PtrType* out_ptr_type() const { return type()->op(1)->as<PtrType>(); }
    const Def* alloced_type() const { return out_ptr_type()->pointee(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Slot;
    friend class World;
};

/// Loads with current effect <tt>mem</tt> from <tt>ptr</tt> to produce a pair of a new effect and the loaded value.
class Load : public MemOp {
private:
    Load(const Def* type, const Def* mem, const Def* ptr, const Def* dbg)
        : MemOp(Node_Load, rebuild, type, {mem, ptr}, 0, dbg)
    {}

public:
    const Def* ptr() const { return op(1); }
    const Def* out_val() const { return out(1); }
    const Sigma* type() const { return MemOp::type()->as<Sigma>(); }
    const Def* out_val_type() const { return type()->op(1); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Load;
    friend class World;
};

/// Stores with current effect <tt>mem</tt> <tt>value</tt> into <tt>ptr</tt> while producing a new effect.
class Store : public MemOp {
private:
    Store(const Def* mem, const Def* ptr, const Def* value, const Def* dbg)
        : MemOp(Node_Store, rebuild, mem->type(), {mem, ptr, value}, 0, dbg)
    {}

public:
    const Def* ptr() const { return op(1); }
    const Def* val() const { return op(2); }
    const MemType* type() const { return MemOp::type()->as<MemType>(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Tag = Tags::Store;
    friend class World;
};

/*
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
    Flags flags() const { return Flags(Def::flags()); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);
    std::ostream& stream_assignment(std::ostream&) const override;

    static constexpr auto Tag = Tags::Assembly;
    friend class World;
};

inline Assembly::Flags operator|(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) | static_cast<int>(rhs)); }
inline Assembly::Flags operator&(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) & static_cast<int>(rhs)); }
inline Assembly::Flags operator|=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs | rhs; }
inline Assembly::Flags operator&=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs & rhs; }
*/

template<class To>
using PrimOpMap     = GIDMap<const PrimOp*, To>;
using PrimOpSet     = GIDSet<const PrimOp*>;
using PrimOp2PrimOp = PrimOpMap<const PrimOp*>;

}

#endif
