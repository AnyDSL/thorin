#ifndef ANYDSL2_JUMPTARGET_H
#define ANYDSL2_JUMPTARGET_H

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

    virtual const DefNode* load() const = 0;
    virtual void store(const DefNode* val) const = 0;
    virtual World& world() const = 0;

    /// Create \p RVal.
    inline static RefPtr create(const DefNode* def);
    /// Create \p VarRef.
    inline static RefPtr create(Lambda* bb, size_t handle, const Type*, const char* name);
    /// Create \p ArrayDeclRef.
    inline static RefPtr create_array(RefPtr lref, const DefNode* index, IRBuilder& builde);
    /// Create \p TupleRef.
    inline static RefPtr create_tuple(RefPtr lref, const DefNode* indexr);
    /// Create \p SlotRef.
    inline static RefPtr create(const Slot* slot, IRBuilder& builder);
};

class RVal : public Ref {
public:
    RVal(const DefNode* def)
        : def_(def)
    {}

    virtual const DefNode* load() const { return def_; }
    virtual void store(const DefNode* val) const { ANYDSL2_UNREACHABLE; }
    virtual World& world() const;

private:
    const DefNode* def_;
};

class VarRef : public Ref {
public:
    VarRef(Lambda* bb, size_t handle, const Type* type, const char* name)
        : bb_(bb)
        , handle_(handle)
        , type_(type)
        , name_(name)
    {}

    virtual const DefNode* load() const;
    virtual void store(const DefNode* def) const;
    virtual World& world() const;

private:
    Lambda* bb_;
    size_t handle_;
    const Type* type_;
    const char* name_;
};

// FIXME: nice integration
class ArrayDeclRef : public Ref {
public:
    ArrayDeclRef(RefPtr lref, const DefNode* index, IRBuilder& builder)
        : lref_(std::move(lref))
        , index_(index)
        , builder_(builder)
    {}

    virtual const DefNode* load() const;
    virtual void store(const DefNode* val) const;
    virtual World& world() const;

private:
    RefPtr lref_;
    const DefNode* index_;
    IRBuilder& builder_;
};

class TupleRef : public Ref {
public:
    TupleRef(RefPtr lref, const DefNode* index)
        : lref_(std::move(lref))
        , index_(index)
        , loaded_(nullptr)
    {}

    virtual const DefNode* load() const;
    virtual void store(const DefNode* val) const;
    virtual World& world() const;

private:
    RefPtr lref_;
    const DefNode* index_;

    /// Caches loaded value to prevent quadratic blow up in calls.
    mutable const DefNode* loaded_;
};

class SlotRef : public Ref {
public:
    SlotRef(const Slot* slot, IRBuilder& builder)
        : slot_(slot)
        , builder_(builder)
    {}

    virtual const DefNode* load() const;
    virtual void store(const DefNode* val) const;
    virtual World& world() const;

private:
    const Slot* slot_;
    IRBuilder& builder_;
};

RefPtr Ref::create(const DefNode* def) { return RefPtr(new RVal(def)); }
RefPtr Ref::create_array(RefPtr lref, const DefNode* index, IRBuilder& builder) { return RefPtr(new ArrayDeclRef(std::move(lref), index, builder)); }
RefPtr Ref::create_tuple(RefPtr lref, const DefNode* index) { return RefPtr(new TupleRef(std::move(lref), index)); }
RefPtr Ref::create(Lambda* bb, size_t handle, const Type* type, const char* name) { 
    return RefPtr(new VarRef(bb, handle, type, name)); 
}
RefPtr Ref::create(const Slot* slot, IRBuilder& builder) { return RefPtr(new SlotRef(slot, builder)); }

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
    void branch(const DefNode* cond, JumpTarget& t, JumpTarget& f);
    void mem_call(const DefNode* to, ArrayRef<const DefNode*> args, const Type* ret_type);
    void tail_call(const DefNode* to, ArrayRef<const DefNode*> args);
    void param_call(const Param* ret_param, ArrayRef<const DefNode*> args);
    const Param* cascading_call(const DefNode* to, ArrayRef<const DefNode*> args, const Type* ret_type);
    const DefNode* get_mem();
    void set_mem(const DefNode* def);

    Lambda* cur_bb;

protected:
    World& world_;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
