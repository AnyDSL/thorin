#ifndef ANYDSL_MEMOP_H
#define ANYDSL_MEMOP_H

#include "anydsl/primop.h"

namespace anydsl {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:

    MemOp(int kind, size_t size, const Type* type, const Def* mem);

public:

    const Def* mem() const { return op(0); }
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:

    Access(int kind, size_t size, const Type* type, const Def* mem, const Def* ptr)
        : MemOp(kind, size, type, mem)
    {
        assert(size >= 2);
        set_op(1, ptr);
    }

public:

    const Def* ptr() const { return op(1); }
};

//------------------------------------------------------------------------------

class Load : public Access {
private:

    Load(const Def* mem, const Def* ptr);

    virtual void vdump(Printer &printer) const;

public:

    const Def* ptr() const { return op(1); }
    const Def* extract_mem() const;
    const Def* extract_val() const;

    virtual Load* stub() const { return new Load(*this); }

private:

    mutable const Def* extract_mem_;
    mutable const Def* extract_val_;

    friend class World;
};

//------------------------------------------------------------------------------

class Store : public Access {
private:

    Store(const Def* mem, const Def* ptr, const Def* val);

    virtual void vdump(Printer &printer) const;

public:

    const Def* val() const { return op(2); }

    virtual Store* stub() const { return new Store(*this); }

    friend class World;
};

//------------------------------------------------------------------------------

class Enter : public MemOp {
private:

    Enter(const Def* mem);

    virtual void vdump(Printer &printer) const;

public:

    const Def* extract_mem() const;
    const Def* extract_frame() const;

    virtual Enter* stub() const { return new Enter(*this); }

private:

    mutable const Def* extract_mem_;
    mutable const Def* extract_frame_;

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:

    Leave(const Def* mem, const Def* frame);

    virtual void vdump(Printer &printer) const;

public:

    virtual Leave* stub() const { return new Leave(*this); }

    const Def* frame() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

/**
 * This represents a slot in a stack frame opend via \p Enter.
 * Loads from this address yield \p Bottom if the frame has already been closed via \p Leave.
 * This \p PrimOp is technically \em not a \p MemOp.
 */
class Slot : public PrimOp {
private:

    Slot(const Def* frame, const Type* type);

    virtual void vdump(Printer &printer) const;

    bool equal(const Node* other) const;
    size_t hash() const;

public:

    virtual Slot* stub() const { return new Slot(*this); }

    const Def* frame() const { return op(0); }

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_MEMOP_H
