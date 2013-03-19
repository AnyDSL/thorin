#ifndef ANYDSL2_MEMOP_H
#define ANYDSL2_MEMOP_H

#include "anydsl2/primop.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class MemOp : public PrimOp {
protected:

    MemOp(size_t size, int kind, const Type* type, const Def* mem, const std::string& name);

public:

    const Def* mem() const { return op(0); }
};

//------------------------------------------------------------------------------

class Access : public MemOp {
protected:

    Access(size_t size, int kind, const Type* type, const Def* mem, const Def* ptr, const std::string& name)
        : MemOp(size, kind, type, mem, name)
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

    Load(const DefTuple2& args, const std::string& name)
        : Access(2, args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
    {}

    virtual void vdump(Printer &printer) const;

public:

    const Def* ptr() const { return op(1); }
    const Def* extract_mem() const;
    const Def* extract_val() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Store : public Access {
private:

    Store(const DefTuple3& args, const std::string& name)
        : Access(3, args.get<0>(), args.get<1>(), args.get<2>(), args.get<3>(), name)
    {
        set_op(2, args.get<4>());
    }

    virtual void vdump(Printer &printer) const;

public:

    const Def* val() const { return op(2); }

    friend class World;
};

//------------------------------------------------------------------------------

class Enter : public MemOp {
private:

    Enter(const DefTuple1& args, const std::string& name)
        : MemOp(1, args.get<0>(), args.get<1>(), args.get<2>(), name)
    {}

    virtual void vdump(Printer &printer) const;

public:

    const Def* extract_mem() const;
    const Def* extract_frame() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Leave : public MemOp {
private:

    Leave(const DefTuple2& args, const std::string& name)
        : MemOp(2, args.get<0>(), args.get<1>(), args.get<2>(), name)
    {
        set_op(1, args.get<3>());
    }

    virtual void vdump(Printer &printer) const;

public:

    const Def* frame() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

typedef boost::tuple<int, const Type*, size_t, const Def*> SlotTuple;

/**
 * This represents a slot in a stack frame opend via \p Enter.
 * Loads from this address yield \p Bottom if the frame has already been closed via \p Leave.
 * This \p PrimOp is technically \em not a \p MemOp.
 */
class Slot : public PrimOp {
private:

    Slot(const SlotTuple& args, const std::string& name) 
        : PrimOp(1, args.get<0>(), args.get<1>(), name)
        , index_(args.get<2>())
    {
        set_op(0, args.get<3>());
    }

    virtual void vdump(Printer &printer) const;

public:

    size_t index() const { return index_; }
    const Def* frame() const { return op(0); }
    ANYDSL2_HASH_EQUAL
    SlotTuple as_tuple() const { return SlotTuple(kind(), type(), index(), frame()); }

private:

    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

class LEA : public PrimOp {
private:

    LEA(const DefTuple2& args, const std::string& name)
        : PrimOp(2, args.get<0>(), args.get<1>(), name)
    {
        set_op(0, args.get<2>());
        set_op(0, args.get<3>());
    }

    virtual void vdump(Printer &printer) const;

public:

    const Def* ptr() const { return op(0); }
    const Def* index() const { return op(1); }

    friend class World;
};

//------------------------------------------------------------------------------

typedef boost::tuple<int, const Type*, const std::string&, const Def*, ArrayRef<const Def*>, bool> CCallTuple;

class CCall : public MemOp {
private:

    CCall(const CCallTuple& args, const std::string& name)
        : MemOp(args.get<4>().size() + 1, args.get<0>(), args.get<1>(), args.get<3>(), name)
        , callee_(args.get<2>())
        , vararg_(args.get<5>())
    {
        size_t x = 1;
        for_all (arg, args.get<4>())
            set_op(x++, arg);
    }

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
    CCallTuple as_tuple() const { 
        return CCallTuple(kind(), type(), callee(), ops().front(), ops().slice_back(1), vararg()); 
    }

private:

    ANYDSL2_HASH_EQUAL

    std::string callee_;
    bool vararg_;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // ANYDSL2_MEMOP_H
