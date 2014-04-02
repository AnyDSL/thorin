#ifndef THORIN_JUMPTARGET_H
#define THORIN_JUMPTARGET_H

#include "thorin/def.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class IRBuilder;
class Lambda;
class Param;
class Ref;
class Slot;
class Type;
class World;

//------------------------------------------------------------------------------

class Var {
public:
    enum Kind {
        Empty,
        ValRef,
        SlotRef
    };

    Var()
        : kind_(Empty)
        , builder_(nullptr)
    {}
    Var(IRBuilder& builder, size_t handle, const Type* type, const char* name);
    Var(IRBuilder& builder, const Slot* slot);

    const Kind kind() const { return kind_; }
    Def load() const;
    void store(Def val) const;
    operator bool() { return kind() != Empty; }

private:
    Kind kind_;
    IRBuilder* builder_;
    union {
        struct {
            size_t handle_;
            const Type* type_;
            const char* name_;
        };
        struct {
            const Slot* slot_;
        };
    };
};

//------------------------------------------------------------------------------

// THIS CODE WILL BE REMOVED

typedef AutoPtr<const Ref> RefPtr;

class Ref {
public:
    Ref(World& world)
        : world_(world)
    {}
    virtual ~Ref() {}

    virtual Def load() const = 0;
    virtual void store(Def val) const = 0;
    World& world() const { return world_; }

    inline static RefPtr create(Def def);                                                   ///< Create \p RVal.
    inline static RefPtr create(Lambda* bb, size_t handle, const Type*, const char* name);  ///< Create \p VarRef.
    inline static RefPtr create(RefPtr lref, Def index);                                    ///< Create \p AggRef.
    inline static RefPtr create(IRBuilder& builder, RefPtr lref, Def index);                ///< Create \p PtrAggRef.
    inline static RefPtr create(IRBuilder& builder, const Slot* slot);                      ///< Create \p SlotRef.

private:
    World& world_;
};

class RVal : public Ref {
public:
    RVal(Def def)
        : Ref(def->world())
        , def_(def)
    {}

    virtual Def load() const { return def_; }
    virtual void store(Def val) const { THORIN_UNREACHABLE; }

private:
    Def def_;
};

class VarRef : public Ref {
public:
    VarRef(Lambda* bb, size_t handle, const Type* type, const char* name);

    virtual Def load() const;
    virtual void store(Def def) const;

private:
    Lambda* bb_;
    size_t handle_;
    const Type* type_;
    const char* name_;
};

class AggRef : public Ref {
public:
    AggRef(RefPtr lref, Def index)
        : Ref(lref->world())
        , lref_(std::move(lref))
        , index_(index)
        , loaded_(nullptr)
    {}

    virtual Def load() const;
    virtual void store(Def val) const;

private:
    RefPtr lref_;
    Def index_;
    mutable Def loaded_; ///< Caches loaded value to prevent quadratic blow up in calls.
};

class SlotRef : public Ref {
public:
    SlotRef(IRBuilder& builder, const Slot* slot);

    virtual Def load() const;
    virtual void store(Def val) const;

private:
    IRBuilder& builder_;
    const Slot* slot_;
};

class AggPtrRef : public Ref {
public:
    AggPtrRef(IRBuilder& builder, RefPtr lref, Def index)
        : Ref(lref->world())
        , builder_(builder)
        , lref_(std::move(lref))
        , index_(index)
    {}

    virtual Def load() const;
    virtual void store(Def val) const;

private:
    IRBuilder& builder_;
    RefPtr lref_;
    Def index_;
};

RefPtr Ref::create(Def def) { return RefPtr(new RVal(def)); }
RefPtr Ref::create(RefPtr lref, Def index) { return RefPtr(new AggRef(std::move(lref), index)); }
RefPtr Ref::create(IRBuilder& builder, RefPtr lref, Def index) { return RefPtr(new AggPtrRef(builder, std::move(lref), index)); }
RefPtr Ref::create(IRBuilder& builder, const Slot* slot) { return RefPtr(new SlotRef(builder, slot)); }
RefPtr Ref::create(Lambda* bb, size_t handle, const Type* type, const char* name) { return RefPtr(new VarRef(bb, handle, type, name)); }

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
    Def get_mem();
    void set_mem(Def def);

    Lambda* cur_bb;

protected:
    World& world_;
};

//------------------------------------------------------------------------------

}

#endif
