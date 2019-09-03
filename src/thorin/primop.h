#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/util.h"

namespace thorin {

//------------------------------------------------------------------------------

/// Reinterprets the bits of <tt>from</tt> as type <tt>to</tt>.
class Bitcast : public Def {
private:
    Bitcast(const Def* to, const Def* from, const Def* dbg)
        : Def(Node, rebuild, to, {from}, 0, dbg)
    {}

public:
    const Def* from() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Bitcast;
    friend class World;
};

/// Data constructor for a @p VariantType.
class Variant : public Def {
private:
    Variant(const VariantType* variant_type, const Def* value, const Def* dbg)
        : Def(Node, rebuild, variant_type, {value}, 0, dbg)
    {
        assert(std::find(variant_type->ops().begin(), variant_type->ops().end(), value->type()) != variant_type->ops().end());
    }

public:
    const VariantType* type() const { return Def::type()->as<VariantType>(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Variant;
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
    LEA(const Def* type, const Def* ptr, const Def* index, const Def* dbg)
        : Def(Node, rebuild, type, {ptr, index}, 0, dbg)
    {}

public:
    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }
    const Ptr* type() const { return Def::type()->as<Ptr>(); }
    const Ptr* ptr_type() const { return ptr()->type()->as<Ptr>(); } ///< Returns the Ptr from @p ptr().
    const Def* ptr_pointee() const { return ptr_type()->pointee(); }        ///< Returns the type referenced by @p ptr().
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::LEA;
    friend class World;
};

/// Casts the underlying @p def to a dynamic value during @p partial_evaluation.
class Hlt : public Def {
private:
    Hlt(const Def* def, const Def* dbg)
        : Def(Node, rebuild, def->type(), {def}, 0, dbg)
    {}

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Hlt;
    friend class World;
};

/// Evaluates to @c true, if @p def is a literal.
class Known : public Def {
private:
    Known(const Def* def, const Def* dbg);

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Known;
    friend class World;
};

/**
 * If a lam typed def is wrapped in @p Run primop, it will be specialized into a callee whenever it is called.
 * Otherwise, this @p Def evaluates to @p def.
 */
class Run : public Def {
private:
    Run(const Def* def, const Def* dbg)
        : Def(Node, rebuild, def->type(), {def}, 0, dbg)
    {}

public:
    const Def* def() const { return op(0); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Run;
    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 */
class Global : public Def {
private:
    Global(const Def* type, const Def* id, const Def* init, bool is_mutable, const Def* dbg)
        : Def(Node, rebuild, type, {id, init}, is_mutable, dbg)
    {}

public:
    /// This thing's sole purpose is to differentiate on global from another.
    const Def* id() const { return op(0); }
    const Def* init() const { return op(1); }
    bool is_mutable() const { return fields(); }
    const Ptr* type() const { return Def::type()->as<Ptr>(); }
    const Def* alloced_type() const { return type()->pointee(); }
    const char* op_name() const override;

    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Global;
    friend class World;
};

/// Allocates memory on the heap.
class Alloc : public Def {
private:
    Alloc(const Def* type, const Def* mem, const Def* dbg)
        : Def(Node, rebuild, type, {mem}, 0, dbg)
    {
        assert(mem->type()->isa<Mem>());
    }

public:
    const Def* mem() const { return op(0); }
    const Def* out_mem() const { return out(0); }
    const Def* out_ptr() const { return out(1); }
    const Sigma* type() const { return Def::type()->as<Sigma>(); }
    const Def* alloced_type() const { return out_ptr()->type()->as<Ptr>()->pointee(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Alloc;
    friend class World;
};

/// Allocates memory on the stack.
/// TODO eventually substitute with Alloc
class Slot : public Def {
private:
    Slot(const Def* type, const Def* mem, const Def* dbg)
        : Def(Node, rebuild, type, {mem}, 0, dbg)
    {
        assert(mem->type()->isa<Mem>());
    }

public:
    const Def* mem() const { return op(0); }
    const Def* out_mem() const { return out(0); }
    const Def* out_ptr() const { return out(1); }
    const Sigma* type() const { return Def::type()->as<Sigma>(); }
    const Def* alloced_type() const { return out_ptr()->type()->as<Ptr>()->pointee(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Slot;
    friend class World;
};

/// Loads with current effect <tt>mem</tt> from <tt>ptr</tt> to produce a pair of a new effect and the loaded value.
class Load : public Def {
private:
    Load(const Def* type, const Def* mem, const Def* ptr, const Def* dbg)
        : Def(Node, rebuild, type, {mem, ptr}, 0, dbg)
    {
        assert(mem->type()->isa<Mem>());
    }

public:
    const Def* mem() const { return op(0); }
    const Def* ptr() const { return op(1); }
    const Def* out_mem() const { return out(0); }
    const Def* out_val() const { return out(1); }
    const Sigma* type() const { return Def::type()->as<Sigma>(); }
    const Def* out_val_type() const { return type()->op(1); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Load;
    friend class World;
};

/// Stores with current effect <tt>mem</tt> <tt>value</tt> into <tt>ptr</tt> while producing a new effect.
class Store : public Def {
private:
    Store(const Def* mem, const Def* ptr, const Def* value, const Def* dbg)
        : Def(Node, rebuild, mem->type(), {mem, ptr, value}, 0, dbg)
    {
        assert(mem->type()->isa<Mem>());
    }

public:
    const Def* mem() const { return op(0); }
    const Def* ptr() const { return op(1); }
    const Def* val() const { return op(2); }
    const Def* out_mem() const { return out(0); }
    const Mem* type() const { return Def::type()->as<Mem>(); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);

    static constexpr auto Node = Node::Store;
    friend class World;
};

/*
class Assembly : public Def {
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
    bool has_sideeffects() const { return fields() & HasSideEffects; }
    bool is_alignstack() const { return fields() & IsAlignStack; }
    bool is_inteldialect() const { return fields() & IsIntelDialect; }
    Flags flags() const { return Flags(fields()); }
    static const Def* rebuild(const Def*, World& to, const Def* type, Defs ops, const Def*);
    std::ostream& stream_assignment(std::ostream&) const override;

    static constexpr auto Node = Node::Assembly;
    friend class World;
};

inline Assembly::Flags operator|(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) | static_cast<int>(rhs)); }
inline Assembly::Flags operator&(Assembly::Flags lhs, Assembly::Flags rhs) { return static_cast<Assembly::Flags>(static_cast<int>(lhs) & static_cast<int>(rhs)); }
inline Assembly::Flags operator|=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs | rhs; }
inline Assembly::Flags operator&=(Assembly::Flags& lhs, Assembly::Flags rhs) { return lhs = lhs & rhs; }
*/

}

#endif
