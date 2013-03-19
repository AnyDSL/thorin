#ifndef ANYDSL2_JUMPTARGET_H
#define ANYDSL2_JUMPTARGET_H

#include <memory>

#include "anydsl2/util/array.h"

namespace anydsl2 {

class Def;
class Lambda;
class Param;
class Ref;
class Slot;
class Type;
class World;

//------------------------------------------------------------------------------

typedef std::auto_ptr<const Ref> RefPtr;

class Ref {
public:

    virtual ~Ref() {}

    virtual const Def* load() const = 0;
    virtual void store(const Def* val) const = 0;
    virtual World& world() const = 0;

    /// Create \p RVal.
    inline static RefPtr create(const Def* def);
    /// Create \p VarRef.
    inline static RefPtr create(Lambda* bb, size_t handle, const Type*, const char* name);
    /// Create \p TupleRef.
    inline static RefPtr create(RefPtr lref, const Def* index);
    /// Create \p SlotRef.
    inline static RefPtr create(const Slot* slot, const Def*& mem);
};

class RVal : public Ref {
public:

    RVal(const Def* def)
        : def_(def)
    {}

    virtual const Def* load() const { return def_; }
    virtual void store(const Def* val) const { ANYDSL2_UNREACHABLE; }
    virtual World& world() const;

private:

    const Def* def_;
};

class VarRef : public Ref {
public:

    VarRef(Lambda* bb, size_t handle, const Type* type, const char* name)
        : bb_(bb)
        , handle_(handle)
        , type_(type)
        , name_(name)
    {}

    virtual const Def* load() const;
    virtual void store(const Def* def) const;
    virtual World& world() const;

private:

    Lambda* bb_;
    size_t handle_;
    const Type* type_;
    const char* name_;
};

class TupleRef : public Ref {
public:

    TupleRef(RefPtr lref, const Def* index)
        : lref_(lref)
        , index_(index)
        , loaded_(0)
    {}

    virtual const Def* load() const;
    virtual void store(const Def* val) const;
    virtual World& world() const;

private:

    RefPtr lref_;
    const Def* index_;

    /// Caches loaded value to prevent quadratic blow up in calls.
    mutable const Def* loaded_;
};

class SlotRef : public Ref {
public:

    SlotRef(const Slot* slot, const Def*& mem)
        : slot_(slot)
        , mem_(mem)
    {}

    virtual const Def* load() const;
    virtual void store(const Def* val) const;
    virtual World& world() const;

private:

    const Slot* slot_;
    const Def*& mem_;
};

RefPtr Ref::create(const Def* def) { return RefPtr(new RVal(def)); }
RefPtr Ref::create(RefPtr lref, const Def* index) { return RefPtr(new TupleRef(lref, index)); }
RefPtr Ref::create(Lambda* bb, size_t handle, const Type* type, const char* name) { 
    return RefPtr(new VarRef(bb, handle, type, name)); 
}
RefPtr Ref::create(const Slot* slot, const Def*& mem) { return RefPtr(new SlotRef(slot, mem)); }

//------------------------------------------------------------------------------

class JumpTarget {
public:

    JumpTarget(const char* name = "")
        : lambda_(0)
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
        : cur_bb(0)
        , world_(world)
    {}

    World& world() const { return world_; }
    bool is_reachable() const { return cur_bb; }
    void set_unreachable() { cur_bb = 0; }
    Lambda* enter(JumpTarget& jt) { return cur_bb = jt.enter(); }
    Lambda* enter_unsealed(JumpTarget& jt) { return cur_bb = jt.enter_unsealed(world_); }
    void jump(JumpTarget& jt);
    void branch(const Def* cond, JumpTarget& t, JumpTarget& f);
    const Param* mem_call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type);
    const Param* call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type);
    void tail_call(const Def* to, ArrayRef<const Def*> args);
    void return0(const Param* ret_param);
    void return1(const Param* ret_param, const Def*);
    void return2(const Param* ret_param, const Def*, const Def*);

    Lambda* cur_bb;

protected:

    World& world_;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
