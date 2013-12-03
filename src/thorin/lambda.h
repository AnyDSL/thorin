#ifndef THORIN_LAMBDA_H
#define THORIN_LAMBDA_H

#include <vector>

#include "thorin/def.h"
#include "thorin/type.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class GenericMap;
class GenericRef;
class Lambda;
class Pi;
class Scope;

typedef std::vector<Lambda*> Lambdas;
typedef std::vector<const Param*> Params;

//------------------------------------------------------------------------------

class Lambda : public DefNode {
public:
    enum {
        Extern      = 1 << 0, ///< Is the function visible in other translation units?
        Accelerator = 1 << 1, ///< Flag for the internal Accelerator-Backend (NVVM / SPIR)
        ArrayInit   = 1 << 2, ///< Flag for the external array intialization
        StencilAr   = 1 << 3, ///< Flag for the external stencil intialization
        Vectorize   = 1 << 4, ///< Flag for the external vectorizer
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
    Lambda(size_t gid, const Pi* pi, Attribute attribute, bool is_sealed, const std::string& name)
        : DefNode(gid, Node_Lambda, 0, pi, true, name)
        , attribute_(attribute)
        , parent_(this)
        , is_sealed_(is_sealed)
        , is_visited_(false)
    {
        params_.reserve(pi->size());
    }
    virtual ~Lambda() { for (auto param : params()) delete param; }

public:
    Lambda* stub(const GenericMap& generic_map) const { return stub(generic_map, name); }
    Lambda* stub(const GenericMap& generic_map, const std::string& name) const;
    Lambda* update_to(Def def) { return update_op(0, def); }
    Lambda* update_op(size_t i, Def def);
    Lambda* update_arg(size_t i, Def def) { return update_op(i+1, def); }
    const Param* append_param(const Type* type, const std::string& name = "");
    Lambdas succs() const;
    Lambdas preds() const;
    const std::vector<const GenericRef*>& generic_refs() const { return generic_refs_; }
    const Params& params() const { return params_; }
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    Def to() const { return op(0); };
    ArrayRef<Def> args() const { return empty() ? ArrayRef<Def>(0, 0) : ops().slice_from_begin(1); }
    Def arg(size_t i) const { return args()[i]; }
    const Pi* pi() const;
    const Pi* to_pi() const;
    const Pi* arg_pi() const;
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
    bool is_connected_to_builtin() const;
    bool is_connected_to_builtin(uint32_t flags) const;
    void dump_head() const;
    void dump_jump() const;
    void destroy_body() { unset_ops(); resize(0); }

    // terminate

    void jump(Def to, ArrayRef<Def> args);
    void branch(Def cond, Def tto, Def fto);
    Lambda* call(Def to, ArrayRef<Def> args, const Type* ret_type);
    Lambda* mem_call(Def to, ArrayRef<Def> args, const Type* ret_type);

    // cps construction

    Def set_value(size_t handle, Def def);
    Def get_value(size_t handle, const Type* type, const char* name = "");
    Def set_mem(Def def);
    Def get_mem();
    Lambda* parent() const { return parent_; }            ///< See \ref parent_ for more information.
    void set_parent(Lambda* parent) { parent_ = parent; } ///< See \ref parent_ for more information.
    void seal();
    bool is_sealed() const { return is_sealed_; }
    void unseal() { is_sealed_ = false; }
    void clear() { values_.clear(); }

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

    Def fix(const Todo& todo);
    Def get_value(const Todo& todo) { return get_value(todo.handle(), todo.type(), todo.name()); }
    Def try_remove_trivial_param(const Param*);
    Def find_def(size_t handle);
    void increase_values(size_t handle) { if (handle >= values_.size()) values_.resize(handle+1); }

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
    std::vector<Def> values_;
    typedef std::vector<Todo> Todos;
    Todos todos_;
    mutable std::vector<const GenericRef*> generic_refs_;

    friend class World;
    friend class Scope;
    friend class GenericRef;
};

//------------------------------------------------------------------------------

class LambdaSet : public std::set<Lambda*, DefNodeLT> {
public:
    typedef std::set<Lambda*, DefNodeLT> Super;

    bool contains(Lambda* def) const { return Super::find(def) != Super::end(); }
    bool visit(Lambda* def) { return !Super::insert(def).second; }
};

template<class Value>
class LambdaMap : public std::map<const Lambda*, Value, DefNodeLT> {
public:
    typedef std::map<const Lambda*, Value, DefNodeLT> Super;
};

template<class Value>
class LambdaMap<Value*> : public std::map<const Lambda*, Value*, DefNodeLT> {
public:
    typedef std::map<const Lambda*, Value*, DefNodeLT> Super;

    Value* find(const Lambda* def) const {
        auto i = Super::find(def);
        return i == Super::end() ? nullptr : i->second;
    }
};

//------------------------------------------------------------------------------

} // namespace thorin

#endif
