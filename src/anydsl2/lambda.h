#ifndef ANYDSL2_LAMBDA_H
#define ANYDSL2_LAMBDA_H

#include <vector>
#include <iostream>

#include "anydsl2/def.h"
#include "anydsl2/type.h"
#include "anydsl2/util/autoptr.h"

namespace anydsl2 {

class GenericMap;
class GenericRef;
class Lambda;
class Pi;
class Scope;

typedef std::vector<Lambda*> Lambdas;
typedef std::vector<const Param*> Params;

class Lambda : public Def {
public:
    enum {
        Extern = 1 << 0, ///< Is the function visible in other translation units?
        Run    = 1 << 1, ///< Flag for the partial evaluator: Evaluate the \em body of this function.
        Cuda   = 1 << 2, ///< Flag for the internal Cuda-Backend
    };

    struct Attribute {
        explicit Attribute(uint32_t flags)
            : flags_(flags)
        {}

        uint32_t filter(uint32_t flags) const { return flags_ & flags; }
        bool is(uint32_t flags) const { return filter(flags) != 0; }
        void set(uint32_t flags) { flags_ |=  flags; }
        void clear(uint32_t flags) { flags_ &= ~flags; }
        void toggle(uint32_t flags) { flags_ ^= flags; }
        uint32_t flags() const { return flags_; }

    private:
        uint32_t flags_;
    };

private:
    Lambda(size_t gid, const Pi* pi, Attribute attribute, bool is_sealed, const std::string& name);
    virtual ~Lambda();

public:
    Lambda* stub(const GenericMap& generic_map) const { return stub(generic_map, name); }
    Lambda* stub(const GenericMap& generic_map, const std::string& name) const;
    Lambda* update_op(size_t i, const Def* def);
    Lambda* update_arg(size_t i, const Def* def) { return update_op(i+1, def); }
    const Param* append_param(const Type* type, const std::string& name = "");
    Lambdas& succs() const;
    Lambdas preds() const;
    Lambdas direct_preds() const;
    const std::vector<const GenericRef*>& generic_refs() const { return generic_refs_; }
    const Params& params() const { return params_; }
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    const Def* to() const { return op(0); };
    ArrayRef<const Def*> args() const { return empty() ? ArrayRef<const Def*>(0, 0) : ops().slice_back(1); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;
    const Pi* to_pi() const;
    const Pi* arg_pi() const;
    size_t sid() const { return sid_; }
    size_t backwards_sid() const { return backwards_sid_; }
    Scope* scope() { return scope_; }
    const Scope* scope() const { return scope_; }
    size_t num_args() const { return args().size(); }
    size_t num_params() const { return params().size(); }
    Attribute& attribute() { return attribute_; }
    const Attribute& attribute() const { return attribute_; }
    /**
     * Is this Lambda part of a call-lambda-cascade? <br>
     * @code
lambda(...) jump (foo, [..., lambda(...) ..., ...]
     * @endcode
     */
    bool is_cascading() const;
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_builtin() const;
    void dump_head() const;
    void dump_jump() const;
    void destroy_body() { unset_ops(); resize(0); }

    // terminate

    void jump(const Def* to, ArrayRef<const Def*> args);
    void branch(const Def* cond, const Def* tto, const Def* fto);
    Lambda* call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type);
    Lambda* mem_call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type);

    // cps construction

    const Def* set_value(size_t handle, const Def* def);
    const Def* get_value(size_t handle, const Type* type, const char* name = "");
    
    Lambda* parent() const { return parent_; }            ///< See \ref parent_ for more information.
    void set_parent(Lambda* parent) { parent_ = parent; } ///< See \ref parent_ for more information.
    void seal();
    bool is_sealed() const { return is_sealed_; }
    void unseal() { is_sealed_ = false; }
    void clear();

private:
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

    const Def* fix(const Todo& todo);
    const Def* get_value(const Todo& todo) { return get_value(todo.handle(), todo.type(), todo.name()); }
    const Def* try_remove_trivial_param(const Param*);
    const Tracker* find_tracker(size_t handle);

    size_t sid_;           ///< \p Scope index, i.e., reverse post-order number.
    size_t backwards_sid_; ///< \p Scope index, i.e., reverse post-order number, while reverting control-flow beginning with the exits.
    Scope* scope_;
    Attribute attribute_;
    Params params_;
    /**
     * There exist three cases to distinguish here.
     * - \p parent_ == this: This \p Lambda is considered as a basic block, i.e., 
     *                       SSA construction will propagate value through this \p Lambda's predecessors.
     * - \p parent_ == nullptr: This \p Lambda is considered as top level function, i.e.,
     *                          SSA construction will stop propagate values here.
     *                          Any \p get_value which arrives here without finding a definition will return \p bottom.
     * - otherwise: This \p Lambda is considered as function head nested in \p parent_.
     *              Any \p get_value which arrives here without finding a definition will recursively try to find one in \p parent_.
     */
    Lambda* parent_;
    bool is_sealed_;
    bool is_visited_;

    typedef std::vector<const Tracker*> TrackedValues;
    TrackedValues tracked_values_;
    typedef std::vector<Todo> Todos;
    Todos todos_;

    mutable Lambdas succs_;
    mutable std::vector<Use> former_uses_;
    mutable std::vector<const Def*> former_ops_;
    mutable std::vector<const GenericRef*> generic_refs_;

    friend class World;
    friend class Scope;
    friend class GenericRef;
};

} // namespace anydsl2

#endif
