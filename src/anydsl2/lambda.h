#ifndef ANYDSL2_LAMBDA_H
#define ANYDSL2_LAMBDA_H

#include <set>
#include <vector>
#include <functional>

#include "anydsl2/def.h"
#include "anydsl2/type.h"
#include "anydsl2/util/autoptr.h"
#include "anydsl2/util/indexmap.h"

namespace anydsl2 {

class GenericMap;
class Lambda;
class Pi;
class Scope;

struct LambdaLT : public std::binary_function<Lambda*, Lambda*, bool> {
    inline bool operator () (Lambda* l1, Lambda* l2) const;
};

typedef std::set<Lambda*, LambdaLT> LambdaSet;
typedef std::vector<Lambda*> Lambdas;

typedef std::vector<const Param*> Params;

struct LambdaAttr {
    enum Attr {
        Extern = 1 << 0,
    };

    explicit LambdaAttr(uint32_t attr)
        : attr(attr)
    {}

    bool is_extern() const { return attr & Extern; }
    void set_extern() { attr |= Extern; }

private:
    uint32_t attr;
};

class Lambda : public Def {
private:

    Lambda(size_t gid, const Pi* pi, LambdaAttr attr, uintptr_t group, bool sealed, const std::string& name);
    virtual ~Lambda();

public:

    Lambda* stub(const GenericMap& generic_map) const { return stub(generic_map, name); }
    Lambda* stub(const GenericMap& generic_map, const std::string& name) const;
    Lambda* update(size_t i, const Def* def);
    const Param* append_param(const Type* type, const std::string& name = "");
    Lambdas succs() const;
    Lambdas preds() const;
    Lambdas direct_succs() const;
    Lambdas direct_preds() const;
    Lambdas group_preds() const;
    const Params& params() const { return params_; }
    const Param* param(size_t i) const { return params_[i]; }
    const Def* to() const { return op(0); };
    ArrayRef<const Def*> args() const { return ops().slice_back(1); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;
    const Pi* to_pi() const;
    const Pi* arg_pi() const;
    size_t gid() const { return gid_; }
    size_t sid() const { return sid_; }
    Scope* scope() { return scope_; }
    const Scope* scope() const { return scope_; }
    size_t num_args() const { return args().size(); }
    size_t num_params() const { return params().size(); }
    LambdaAttr& attr() { return attr_; }
    const LambdaAttr& attr() const { return attr_; }
    /**
     * Is this Lambda part of a call-lambda-cascade? <br>
     * @code
lambda(...) jump (foo, [..., lambda(...) ..., ...]
     * @endcode
     */
    bool is_cascading() const;
    bool is_returning() const;
    bool is_bb() const { return order() == 1; }
    bool sid_valid() { return sid_ != size_t(-1); }
    bool sid_invalid() { return sid_ == size_t(-1); }
    void invalidate_sid() { sid_ = size_t(-1); }
    void dump(bool fancy = false, int indent = 0) const;

    // terminate

    void jump(const Def* to, ArrayRef<const Def*> args);
    void jump(const Def* to, ArrayRef<const Def*> args, const Def* arg);
    void jump0(const Def* to) {
        return jump(to, ArrayRef<const Def*>(0, 0));
    }
    void jump1(const Def* to, const Def* arg1) {
        const Def* args[1] = { arg1 };
        return jump(to, args);
    }
    void jump2(const Def* to, const Def* arg1, const Def* arg2) {
        const Def* args[2] = { arg1, arg2 };
        return jump(to, args);
    }
    void jump3(const Def* to, const Def* arg1, const Def* arg2, const Def* arg3) {
        const Def* args[3] = { arg1, arg2, arg3 };
        return jump(to, args);
    }
    void branch(const Def* cond, const Def* tto, const Def* fto);
    Lambda* call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type);
    Lambda* call0(const Def* to, const Type* ret_type) {
        return call(to, ArrayRef<const Def*>(0, 0), ret_type);
    }
    Lambda* call1(const Def* to, const Def* arg1, const Type* ret_type) {
        const Def* args[1] = { arg1 };
        return call(to, args, ret_type);
    }
    Lambda* call2(const Def* to, const Def* arg1, const Def* arg2, const Type* ret_type) {
        const Def* args[2] = { arg1, arg2 };
        return call(to, args, ret_type);
    }
    Lambda* call3(const Def* to, const Def* arg1, const Def* arg2, const Def* arg3, const Type* ret_type) {
        const Def* args[3] = { arg1, arg2, arg3 };
        return call(to, args, ret_type);
    }

    // cps construction

    const Def* set_value(size_t handle, const Def* def) { return defs_[handle] = def; }
    const Def* get_value(size_t handle, const Type* type, const char* name = "");
    Lambda* parent() const { return parent_; }
    void set_parent(Lambda* parent) { parent_ = parent; }
    void seal();
    bool sealed() const { return sealed_; }
    uintptr_t group() const { return group_; }
    void set_group(uintptr_t group) { group_ = group; }

private:

    virtual void vdump(Printer& printer) const;

    class Todo {
    public:

        Todo() {}
        Todo(size_t handle, size_t index, const Type* type, const char* name)
            : handle_(handle)
            , index_(index)
            , type_(type)
            , name_(name)
        {}

        size_t handle() const { return handle_; }
        size_t index() const { return index_; }
        const Type* type() const { return type_; }
        const char* name() const { return name_; }

    private:

        size_t handle_;
        size_t index_;
        const Type* type_;
        const char* name_;
    };

    void fix(Todo todo);

    size_t gid_; ///< global index
    size_t sid_; ///< scope index
    union {
        Scope* scope_;
        uintptr_t group_;
    };
    LambdaAttr attr_;
    Params params_;
    Lambda* parent_;
    bool sealed_;

    typedef IndexMap<const Def> DefMap;
    DefMap defs_;

    typedef std::vector<Todo> Todos;
    Todos todos_;

    friend class World;
    friend class Scope;
};

bool LambdaLT::operator () (Lambda* l1, Lambda* l2) const { return l1->gid() < l2->gid(); };

} // namespace anydsl2

#endif
