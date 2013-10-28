#ifndef ANYDSL2_MEMOP_H
#define ANYDSL2_MEMOP_H

#include "anydsl2/primop.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:
    MemOp(size_t size, NodeKind kind, const Type* type, Def mem, const std::string& name);

public:
    Def mem() const { return op(0); }
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:
    Access(size_t size, NodeKind kind, const Type* type, Def mem, Def ptr, const std::string& name)
        : MemOp(size, kind, type, mem, name)
    {
        assert(size >= 2);
        set_op(1, ptr);
    }

public:
    Def ptr() const { return op(1); }
};

//------------------------------------------------------------------------------

class Load : public Access {
private:
    Load(Def mem, Def ptr, const std::string& name);

public:
    Def ptr() const { return op(1); }
    Def extract_mem() const;
    Def extract_val() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Store : public Access {
private:
    Store(Def mem, Def ptr, Def value, const std::string& name);

public:
    Def val() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Enter : public MemOp {
private:
    Enter(Def mem, const std::string& name);

public:
    Def extract_mem() const;
    Def extract_frame() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:
    Leave(Def mem, Def frame, const std::string& name);

public:
    Def frame() const { return op(1); }

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
    Slot(const Type* type, Def frame, size_t index, const std::string& name);

public:
    Def frame() const { return op(0); }
    size_t index() const { return index_; }

    virtual size_t hash() const { return hash_combine(PrimOp::hash(), index()); }
    virtual bool equal(const PrimOp* other) const {
        return PrimOp::equal(other) ? this->index() == other->as<Slot>()->index() : false;
    }

private:
    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

class LEA : public PrimOp {
private:
    LEA(Def ptr, Def index, const std::string& name);

public:
    Def ptr() const { return op(0); }
    Def index() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // ANYDSL2_MEMOP_H
