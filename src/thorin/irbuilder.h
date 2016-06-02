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
    enum Kind {
        Empty,
        ImmutableValRef,
        MutableValRef,
        PtrRef,
        AggRef,
    };

    Value()
        : kind_(Empty)
        , builder_(nullptr)
        , handle_(-1)
        , type_(nullptr)
        , name_(nullptr)
        , def_(nullptr)
    {}
    Value(const Value& value)
        : kind_   (value.kind())
        , builder_(value.builder_)
        , handle_ (value.handle_)
        , type_   (value.type_)
        , name_   (value.name_)
        , def_    (value.def_)
        , value_  (value.value_ == nullptr ? nullptr : new Value(*value.value_))
    {}
    Value(Value&& value)
        : Value()
    {
        swap(*this, value);
    }

    Value static create_val(IRBuilder&, const Def* val);
    Value static create_mut(IRBuilder&, size_t handle, const Type* type, const char* name);
    Value static create_ptr(IRBuilder&, const Def* ptr);
    Value static create_agg(Value value, const Def* offset);

    Kind kind() const { return kind_; }
    IRBuilder* builder() const { return builder_; }
    World& world() const;
    const Def* load(const Location& loc) const;
    void store(const Def* val, const Location& loc) const;
    const Def* def() const { return def_; }
    operator bool() { return kind() != Empty; }
    bool use_lea() const;

    Value& operator= (Value other) { swap(*this, other); return *this; }
    friend void swap(Value& v1, Value& v2) {
        using std::swap;
        swap(v1.kind_,    v2.kind_);
        swap(v1.builder_, v2.builder_);
        swap(v1.handle_,  v2.handle_);
        swap(v1.type_,    v2.type_);
        swap(v1.name_,    v2.name_);
        swap(v1.def_,     v2.def_);
        swap(v1.value_,     v2.value_);
    }

private:
    Kind kind_;
    IRBuilder* builder_;
    size_t handle_;
    const Type* type_;
    const char* name_;
    const Def* def_;
    std::unique_ptr<Value> value_;
};

//------------------------------------------------------------------------------

class JumpTarget : public HasLocation {
public:
    JumpTarget(const Location& loc, const char* name = "")
        : HasLocation(loc)
        , continuation_(nullptr)
        , first_(false)
        , name_(name)
    {}
#ifndef NDEBUG
#else
    ~JumpTarget();
#endif

    World& world() const { assert(continuation_); return continuation_->world(); }
    void seal() { assert(continuation_); continuation_->seal(); }

private:
    void jump_from(Continuation* bb);
    Continuation* branch_to(World& world, const Location& loc);
    Continuation* untangle();
    Continuation* enter();
    Continuation* enter_unsealed(World& world);

    Continuation* continuation_;
    bool first_;
    const char* name_;

    friend void Continuation::jump(JumpTarget&, const Location&);
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
    const Def* create_frame(const Location& loc);
    const Def* alloc(const Type* type, const Def* extra, const Location& loc, const std::string& name = "");
    const Def* load(const Def* ptr, const Location& loc, const std::string& name = "");
    const Def* extract(const Def* agg, const Def* index, const Location& loc, const std::string& name = "");
    const Def* extract(const Def* agg, u32 index, const Location& loc, const std::string& name = "");
    void store(const Def* ptr, const Def* val, const Location& loc, const std::string& name = "");
    Continuation* enter(JumpTarget& jt) { return cur_bb = jt.enter(); }
    void enter(Continuation* continuation) { cur_bb = continuation; continuation->seal(); }
    Continuation* enter_unsealed(JumpTarget& jt) { return cur_bb = jt.enter_unsealed(world_); }
    void jump(JumpTarget& jt, const Location& loc);
    void branch(const Def* cond, JumpTarget& t, JumpTarget& f, const Location& loc);
    const Def* call(const Def* to, Types type_args, Defs args, const Type* ret_type, const Location& loc);
    const Def* get_mem();
    void set_mem(const Def* def);
    Continuation* continuation(const FnType* fn, const Location& loc, CC cc = CC::C, Intrinsic intrinsic = Intrinsic::None, const std::string& name = "");
    Continuation* continuation(const FnType* fn, const Location& loc, const std::string& name) { return continuation(fn, loc, CC::C, Intrinsic::None, name); }
    Continuation* continuation(const Location& loc, const std::string& name);

    Continuation* cur_bb;

protected:
    World& world_;
};

//------------------------------------------------------------------------------

}

#endif
