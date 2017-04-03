#ifndef THORIN_IRBUILDER_H
#define THORIN_IRBUILDER_H

#include <memory>

#include "thorin/def.h"
#include "thorin/continuation.h"
#include "thorin/util/array.h"

namespace thorin {

class IRBuilder;
class Continuation;
class Slot;
class World;

//------------------------------------------------------------------------------

class Value {
public:
    enum Tag {
        Empty,
        ImmutableValRef,
        MutableValRef,
        PtrRef,
        AggRef,
    };

    Value()
        : tag_(Empty)
        , builder_(nullptr)
        , handle_(-1)
        , type_(nullptr)
        , def_(nullptr)
    {}
    Value(const Value& value)
        : tag_   (value.tag())
        , builder_(value.builder_)
        , handle_ (value.handle_)
        , type_   (value.type_)
        , def_    (value.def_)
        , value_  (value.value_ == nullptr ? nullptr : new Value(*value.value_))
    {}
    Value(Value&& value)
        : Value()
    {
        swap(*this, value);
    }

    Value static create_val(IRBuilder&, const Def* val);
    Value static create_mut(IRBuilder&, size_t handle, const Type* type);
    Value static create_ptr(IRBuilder&, const Def* ptr);
    Value static create_agg(Value value, const Def* offset);

    Tag tag() const { return tag_; }
    IRBuilder* builder() const { return builder_; }
    World& world() const;
    const Def* load(Debug) const;
    void store(const Def* val, Debug) const;
    const Def* def() const { return def_; }
    operator bool() { return tag() != Empty; }
    bool use_lea() const;

    Value& operator= (Value other) { swap(*this, other); return *this; }
    friend void swap(Value& v1, Value& v2) {
        using std::swap;
        swap(v1.tag_,     v2.tag_);
        swap(v1.builder_, v2.builder_);
        swap(v1.handle_,  v2.handle_);
        swap(v1.type_,    v2.type_);
        swap(v1.def_,     v2.def_);
        swap(v1.value_,   v2.value_);
    }

private:
    Tag tag_;
    IRBuilder* builder_;
    size_t handle_;
    const Type* type_;
    const Def* def_;
    std::unique_ptr<Value> value_;
};

//------------------------------------------------------------------------------

class JumpTarget {
public:
    JumpTarget(Debug dbg)
        : debug_(dbg)
    {}
#ifndef NDEBUG
#else
    ~JumpTarget();
#endif

    const Debug& debug() const { return debug_; }
    const std::string& name() const { return debug().name(); }
    World& world() const { assert(continuation_); return continuation_->world(); }
    void seal() { assert(continuation_); continuation_->seal(); }

private:
    void jump_from(Continuation* bb);
    Continuation* branch_to(World& world, Debug);
    Continuation* untangle();
    Continuation* enter();
    Continuation* enter_unsealed(World& world);

    Debug debug_;
    Continuation* continuation_ = nullptr;
    bool first_ = false;

    friend void Continuation::jump(JumpTarget&, Debug);
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
    const Def* create_frame(Debug);
    const Def* alloc(const Type* type, const Def* extra, Debug dbg = {});
    const Def* load(const Def* ptr, Debug dbg = {});
    const Def* extract(const Def* agg, const Def* index, Debug dbg = {});
    const Def* extract(const Def* agg, u32 index, Debug dbg = {});
    void store(const Def* ptr, const Def* val, Debug dbg = {});
    Continuation* enter(JumpTarget& jt) { return cur_bb = jt.enter(); }
    void enter(Continuation* continuation) { cur_bb = continuation; continuation->seal(); }
    Continuation* enter_unsealed(JumpTarget& jt) { return cur_bb = jt.enter_unsealed(world_); }
    void jump(JumpTarget& jt, Debug dbg = {});
    void branch(const Def* cond, JumpTarget& t, JumpTarget& f, Debug dbg = {});
    const Def* call(const Def* to, Defs args, const Type* ret_type, Debug dbg = {});
    const Def* get_mem();
    void set_mem(const Def* def);
    Continuation* continuation(const FnType* fn, CC cc = CC::C, Intrinsic intrinsic = Intrinsic::None, Debug dbg = {});
    Continuation* continuation(const FnType* fn, Debug dbg = {}) { return continuation(fn, CC::C, Intrinsic::None, dbg); }
    Continuation* continuation(Debug dbg = {});

    Continuation* cur_bb;

protected:
    World& world_;
};

//------------------------------------------------------------------------------

}

#endif
