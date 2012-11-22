#ifndef ANYDSL2_MEMOP_H
#define ANYDSL2_MEMOP_H

#include "anydsl2/primop.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:

    MemOp(int kind, size_t size, const Type* type, const Def* mem, const std::string& name);

public:

    const Def* mem() const { return op(0); }
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:

    Access(int kind, size_t size, const Type* type, const Def* mem, const Def* ptr, const std::string& name)
        : MemOp(kind, size, type, mem, name)
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

    Load(const Def* mem, const Def* ptr, const std::string& name);

    virtual void vdump(Printer &printer) const;

public:

    const Def* ptr() const { return op(1); }
    const Def* extract_mem() const;
    const Def* extract_val() const;

private:

    mutable const Def* extract_mem_;
    mutable const Def* extract_val_;

    friend class World;
};

//------------------------------------------------------------------------------

class Store : public Access {
private:

    Store(const Def* mem, const Def* ptr, const Def* val, const std::string& name);

    virtual void vdump(Printer &printer) const;

public:

    const Def* val() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Enter : public MemOp {
private:

    Enter(const Def* mem, const std::string& name);

    virtual void vdump(Printer &printer) const;

public:

    const Def* extract_mem() const;
    const Def* extract_frame() const;

private:

    mutable const Def* extract_mem_;
    mutable const Def* extract_frame_;

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:

    Leave(const Def* mem, const Def* frame, const std::string& name);

    virtual void vdump(Printer &printer) const;

public:

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

    Slot(const Def* frame, const Type* type, const std::string& name);

    virtual void vdump(Printer &printer) const;

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;

public:

    const Def* frame() const { return op(0); }

    friend class World;
};

//------------------------------------------------------------------------------

class CCall : public MemOp {
private:

    CCall(const Def* mem, const std::string& callee, 
          ArrayRef<const Def*> args, const Type* rettype, bool vararg, const std::string& name);

    virtual void vdump(Printer &printer) const;

public:

    bool returns_void() const;
    const Def* extract_mem() const;
    const Def* extract_retval() const;
    const std::string& callee() const { return callee_; }
    bool vararg() const { return vararg_; }
    const Type* rettype() const;
    ArrayRef<const Def*> args() const { return ops().slice_back(1); }
    size_t num_args() const { return args().size(); }

private:

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;

    mutable const Def* extract_mem_;
    mutable const Def* extract_retval_;

    std::string callee_;
    bool vararg_;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // ANYDSL2_MEMOP_H
