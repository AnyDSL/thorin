#ifndef THORIN_MEMOP_H
#define THORIN_MEMOP_H

#include "thorin/primop.h"
#include "thorin/type.h"

namespace thorin {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:
    MemOp(size_t size, NodeKind kind, Type type, Def mem, const std::string& name);

public:
    Def mem() const { return op(0); }
    virtual bool has_mem_out() const { return false; }
    virtual Def mem_out() const { return Def(); }
};

class Alloc : public MemOp {
private:
    Alloc(Def mem, Type type, Def extra, const std::string& name);

public:
    Def extra() const { return op(1); }
    Type alloced_type() const { return type().as<PtrType>()->referenced_type(); }

    friend class World;
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:
    Access(size_t size, NodeKind kind, Type type, Def mem, Def ptr, const std::string& name)
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
    virtual bool has_mem_out() const { return true; }
    virtual Def mem_out() const override { return this; }

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
    virtual bool has_mem_out() const { return true; }
    virtual Def mem_out() const override { return this; }

    friend class World;
};

//------------------------------------------------------------------------------

class MapOp : public MemOp {
protected:
    MapOp(size_t size, NodeKind kind, Type type,
          Def mem, Def ptr, int32_t device, AddressSpace addr_space, const std::string &name);

public:
    Def ptr() const { return op(1); }
    PtrType ptr_type() const { return type().as<TupleType>()->arg(1).as<PtrType>(); }
    AddressSpace addr_space() const { return ptr_type()->addr_space(); }
    int32_t device() const { return ptr_type()->device(); }
};

class Map : public MapOp {
private:
    Map(Def mem, Def ptr, int32_t device, AddressSpace addr_space,
        Def offset, Def size, const std::string &name);

public:
    Def extract_mem() const;
    Def extract_mapped_ptr() const;
    Def mem_offset() const { return op(2); }
    Def mem_size() const { return op(3); }
    virtual bool has_mem_out() const { return true; }
    virtual Def mem_out() const override;

    friend class World;
};

class Unmap : public MapOp {
private:
    Unmap(Def mem, Def ptr, int32_t device, AddressSpace addr_space, const std::string &name);
    virtual bool has_mem_out() const { return true; }
    virtual Def mem_out() const override { return this; }

    friend class World;
};

//------------------------------------------------------------------------------

}

#endif
