#ifndef THORIN_PRIMOP_H
#define THORIN_PRIMOP_H

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/util.h"

namespace thorin {

//------------------------------------------------------------------------------

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

}

#endif
