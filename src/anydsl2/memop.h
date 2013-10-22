#ifndef ANYDSL2_MEMOP_H
#define ANYDSL2_MEMOP_H

#include "anydsl2/primop.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:
    MemOp(size_t size, int kind, const Type* type, const DefNode* mem, const std::string& name);

public:
    const DefNode* mem() const { return op(0); }
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:
    Access(size_t size, int kind, const Type* type, const DefNode* mem, const DefNode* ptr, const std::string& name)
        : MemOp(size, kind, type, mem, name)
    {
        assert(size >= 2);
        set_op(1, ptr);
    }

public:
    const DefNode* ptr() const { return op(1); }
};

//------------------------------------------------------------------------------

class Load : public Access {
private:
    Load(const DefNode* mem, const DefNode* ptr, const std::string& name);

public:
    const DefNode* ptr() const { return op(1); }
    const DefNode* extract_mem() const;
    const DefNode* extract_val() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Store : public Access {
private:
    Store(const DefNode* mem, const DefNode* ptr, const DefNode* value, const std::string& name);

public:
    const DefNode* val() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Enter : public MemOp {
private:
    Enter(const DefNode* mem, const std::string& name);

public:
    const DefNode* extract_mem() const;
    const DefNode* extract_frame() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:
    Leave(const DefNode* mem, const DefNode* frame, const std::string& name);

public:
    const DefNode* frame() const { return op(1); }

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
    Slot(const Type* type, const DefNode* frame, size_t index, const std::string& name);

public:
    const DefNode* frame() const { return op(0); }
    size_t index() const { return index_; }

    virtual size_t hash() const { return hash_combine(PrimOp::hash(), index()); }
    virtual bool equal(const Node* other) const {
        return PrimOp::equal(other) ? this->index() == other->as<Slot>()->index() : false;
    }

private:
    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

class LEA : public PrimOp {
private:
    LEA(const DefNode* ptr, const DefNode* index, const std::string& name);

public:
    const DefNode* ptr() const { return op(0); }
    const DefNode* index() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // ANYDSL2_MEMOP_H
