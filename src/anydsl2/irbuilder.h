#ifndef ANYDSL2_JUMPTARGET_H
#define ANYDSL2_JUMPTARGET_H

#include "anydsl2/def.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/autoptr.h"

namespace anydsl2 {

class DefNode;
class IRBuilder;
class Lambda;
class Param;
class Ref;
class Slot;
class Type;
class World;

//------------------------------------------------------------------------------

typedef AutoPtr<const Ref> RefPtr;

class Ref {
public:
    virtual ~Ref() {}

    virtual Def load() const = 0;
    virtual void store(Def val) const = 0;
    virtual World& world() const = 0;

    /// Create \p RVal.
    inline static RefPtr create(Def def);
    /// Create \p VarRef.
    inline static RefPtr create(Lambda* bb, size_t handle, const Type*, const char* name);
    /// Create \p ArrayValueRef.
    inline static RefPtr create_array_val(RefPtr lref, Def index);
    /// Create \p TupleRef.
    inline static RefPtr create_tuple(RefPtr lref, Def index);
    /// Create \p ArrayPtrRef.
    inline static RefPtr create_array_ptr(IRBuilder& builder, RefPtr lref, Def index);
    /// Create \p SlotRef.
    inline static RefPtr create(IRBuilder& builder, const Slot* slot);
};

class RVal : public Ref {
public:
    RVal(Def def)
        : def_(def)
    {}

    virtual Def load() const { return def_; }
    virtual void store(Def val) const { ANYDSL2_UNREACHABLE; }
    virtual World& world() const;

private:
    Def def_;
};

class VarRef : public Ref {
public:
    VarRef(Lambda* bb, size_t handle, const Type* type, const char* name)
        : bb_(bb)
        , handle_(handle)
        , type_(type)
        , name_(name)
    {}

    virtual Def load() const;
    virtual void store(Def def) const;
    virtual World& world() const;

private:
    Lambda* bb_;
    size_t handle_;
    const Type* type_;
    const char* name_;
};

class ArrayValueRef : public Ref {
public:
    ArrayValueRef(RefPtr lref, Def index)
        : lref_(std::move(lref))
        , index_(index)
        , loaded_(nullptr)
    {}

    virtual Def load() const;
    virtual void store(Def val) const;
    virtual World& world() const;

private:
    RefPtr lref_;
    Def index_;

    /// Caches loaded value to prevent quadratic blow up in calls.
    mutable Def loaded_;
};

class ArrayPtrRef : public Ref {
public:
    ArrayPtrRef(IRBuilder& builder, RefPtr lref, Def index)
        : builder_(builder)
        , lref_(std::move(lref))
        , index_(index)
    {}

    virtual Def load() const;
    virtual void store(Def val) const;
    virtual World& world() const;

private:
    IRBuilder& builder_;
    RefPtr lref_;
    Def index_;
};

class TupleRef : public Ref {
public:
    TupleRef(RefPtr lref, Def index)
        : lref_(std::move(lref))
        , index_(index)
        , loaded_(nullptr)
    {}

    virtual Def load() const;
    virtual void store(Def val) const;
    virtual World& world() const;

private:
    RefPtr lref_;
    Def index_;

    /// Caches loaded value to prevent quadratic blow up in calls.
    mutable Def loaded_;
};

class SlotRef : public Ref {
public:
    SlotRef(IRBuilder& builder, const Slot* slot)
        : builder_(builder)
        , slot_(slot)
    {}

    virtual Def load() const;
    virtual void store(Def val) const;
    virtual World& world() const;

private:
    IRBuilder& builder_;
    const Slot* slot_;
};

RefPtr Ref::create(Def def) { return RefPtr(new RVal(def)); }
RefPtr Ref::create_array_val(RefPtr lref, Def index) { return RefPtr(new ArrayValueRef(std::move(lref), index)); }
RefPtr Ref::create_array_ptr(IRBuilder& builder, RefPtr lref, Def index) { 
    return RefPtr(new ArrayPtrRef(builder, std::move(lref), index)); }
RefPtr Ref::create_tuple(RefPtr lref, Def index) { return RefPtr(new TupleRef(std::move(lref), index)); }
RefPtr Ref::create(IRBuilder& builder, const Slot* slot) { return RefPtr(new SlotRef(builder, slot)); }
RefPtr Ref::create(Lambda* bb, size_t handle, const Type* type, const char* name) { 
    return RefPtr(new VarRef(bb, handle, type, name)); 
}

//------------------------------------------------------------------------------

class JumpTarget {
public:
    JumpTarget(const char* name = "")
        : lambda_(nullptr)
        , first_(false)
        , name_(name)
    {}
#ifndef NDEBUG
    ~JumpTarget();
#endif

    World& world() const;
    void seal();
    void jump_from(Lambda* bb);

private:
    Lambda* get(World& world);
    Lambda* untangle();
    Lambda* enter();
    Lambda* enter_unsealed(World& world);

    Lambda* lambda_;
    bool first_;
    const char* name_;

    friend class IRBuilder;
};

//------------------------------------------------------------------------------

class IRBuilder {
public:
    IRBuilder(World& world)
        : cur_bb(nullptr)
        , world_(world)
    {}

    World& world() const { return world_; }
    bool is_reachable() const { return cur_bb != nullptr; }
    void set_unreachable() { cur_bb = nullptr; }
    Lambda* enter(JumpTarget& jt) { return cur_bb = jt.enter(); }
    Lambda* enter_unsealed(JumpTarget& jt) { return cur_bb = jt.enter_unsealed(world_); }
    void jump(JumpTarget& jt);
    void branch(Def cond, JumpTarget& t, JumpTarget& f);
    void mem_call(Def to, ArrayRef<Def> args, const Type* ret_type);
    void tail_call(Def to, ArrayRef<Def> args);
    void param_call(const Param* ret_param, ArrayRef<Def> args);
    const Param* cascading_call(Def to, ArrayRef<Def> args, const Type* ret_type);
    Def get_mem();
    void set_mem(Def def);

    Lambda* cur_bb;

protected:
    World& world_;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
