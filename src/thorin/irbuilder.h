#ifndef THORIN_IRBUILDER_H
#define THORIN_IRBUILDER_H

#include <memory>

#include "thorin/def.h"
#include "thorin/lambda.h"
#include "thorin/util/array.h"

namespace thorin {

class IRBuilder;
class Lambda;
class Slot;
class World;

//------------------------------------------------------------------------------

class Var {
public:
    enum Kind {
        Empty,
        ImmutableValRef,
        MutableValRef,
        PtrRef,
        AggRef,
    };

    Var()
        : kind_(Empty)
        , builder_(nullptr)
        , handle_(-1)
        , type_(nullptr)
        , name_(nullptr)
        , def_(nullptr)
    {}
    Var(const Var& var)
        : kind_   (var.kind())
        , builder_(var.builder_)
        , handle_ (var.handle_)
        , type_   (var.type_)
        , name_   (var.name_)
        , def_    (var.def_)
        , var_    (var.var_ == nullptr ? nullptr : new Var(*var.var_))
    {}
    Var(Var&& var)
        : Var()
    {
        swap(*this, var);
    }

    Var static create_val(IRBuilder&, Def val);
    Var static create_mut(IRBuilder&, size_t handle, Type type, const char* name);
    Var static create_ptr(IRBuilder&, Def ptr);
    Var static create_agg(Var var, Def offset);

    Kind kind() const { return kind_; }
    IRBuilder* builder() const { return builder_; }
    World& world() const;
    Def load(const Location& loc) const;
    void store(Def val, const Location& loc) const;
    Def def() const { return def_; }
    operator bool() { return kind() != Empty; }
    bool use_lea() const;

    Var& operator= (Var other) { swap(*this, other); return *this; }
    friend void swap(Var& v1, Var& v2) {
        using std::swap;
        swap(v1.kind_,    v2.kind_);
        swap(v1.builder_, v2.builder_);
        swap(v1.handle_,  v2.handle_);
        swap(v1.type_,    v2.type_);
        swap(v1.name_,    v2.name_);
        swap(v1.def_,     v2.def_);
        swap(v1.var_,     v2.var_);
    }

private:
    Kind kind_;
    IRBuilder* builder_;
    size_t handle_;
    const TypeNode* type_;
    const char* name_;
    const DefNode* def_;
    std::unique_ptr<Var> var_;
};

//------------------------------------------------------------------------------

class JumpTarget {
public:
    JumpTarget(const char* name = "")
        : lambda_(nullptr)
        , first_(false)
        , name_(name)
    {}
#ifndef NDEBUG
#else
    ~JumpTarget();
#endif

    World& world() const { assert(lambda_); return lambda_->world(); }
    void seal() { assert(lambda_); lambda_->seal(); }

private:
    void jump_from(Lambda* bb);
    Lambda* branch_to(World& world, const Location& loc);
    Lambda* untangle();
    Lambda* enter();
    Lambda* enter_unsealed(World& world, const Location& loc);

    Lambda* lambda_;
    bool first_;
    const char* name_;

    friend void Lambda::jump(JumpTarget&);
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
    Def create_frame(const Location& loc);
    Def alloc(Type type, Def extra, const Location& loc, const std::string& name = "");
    Def load(Def ptr, const Location& loc, const std::string& name = "");
    Def extract(Def agg, Def index, const Location& loc, const std::string& name = "");
    Def extract(Def agg, u32 index, const Location& loc, const std::string& name = "");
    void store(Def ptr, Def val, const Location& loc, const std::string& name = "");
    Lambda* enter(JumpTarget& jt) { return cur_bb = jt.enter(); }
    Lambda* enter_unsealed(JumpTarget& jt, const Location& loc) { return cur_bb = jt.enter_unsealed(world_, loc); }
    void jump(JumpTarget& jt);
    void branch(Def cond, JumpTarget& t, JumpTarget& f);
    Def call(ArrayRef<Type> type_args, Def to, ArrayRef<Def> args, Type ret_type);
    Def get_mem();
    void set_mem(Def def);

    Lambda* cur_bb;

protected:
    World& world_;
};

//------------------------------------------------------------------------------

}

#endif
