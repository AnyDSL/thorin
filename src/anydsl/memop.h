#ifndef ANYDSL_MEMOP_H
#define ANYDSL_MEMOP_H

#include "anydsl/primop.h"

namespace anydsl {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:

    MemOp(int kind, const Type* type, size_t size, const Def* mem)
        : PrimOp(kind, type, size)
    {
        assert(size >= 1);
        set_op(0, mem);
    }

public:

    const Def* mem() const { return op(0); }
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:

    Access(int kind, const Type* type, size_t size, const Def* mem, const Def* ptr)
        : MemOp(kind, type, size, mem)
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

    virtual Load* clone() const { return new Load(*this); }

    friend class World;
};

//------------------------------------------------------------------------------

class Store : public Access {
private:

    Store(const Def* mem, const Def* ptr, const Def* val);

    virtual void vdump(Printer &printer) const;

public:

    const Def* val() const { return op(2); }

    virtual Store* clone() const { return new Store(*this); }

    friend class World;
};

//------------------------------------------------------------------------------

class Enter : public MemOp {
private:

    Enter(const Def* mem);

    virtual void vdump(Printer &printer) const;

public:

    virtual Enter* clone() const { return new Enter(*this); }

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:

    Leave(const Def* mem, const Enter* enter);

    virtual void vdump(Printer &printer) const;

public:

    virtual Leave* clone() const { return new Leave(*this); }

    const Enter* enter() const { return op(1)->as<Enter>(); }

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

    Slot(const Enter* enter, const Type* type);

    virtual void vdump(Printer &printer) const;

    bool equal(const Def* other) const;
    size_t hash() const;

public:

    virtual Slot* clone() const { return new Slot(*this); }

    const Enter* enter() const { return op(0)->as<Enter>(); }

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_MEMOP_H
