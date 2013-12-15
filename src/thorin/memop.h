#ifndef THORIN_MEMOP_H
#define THORIN_MEMOP_H

#include "thorin/primop.h"

namespace thorin {

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

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:
    Leave(Def mem, Def frame, const std::string& name);

public:
    const Enter* frame() const { return op(1)->as<Enter>(); }

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace thorin

#endif // THORIN_MEMOP_H
