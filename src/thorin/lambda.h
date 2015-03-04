#ifndef THORIN_LAMBDA_H
#define THORIN_LAMBDA_H

#include <list>
#include <vector>

#include "thorin/def.h"
#include "thorin/type.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class JumpTarget;
class Lambda;
class Scope;

typedef std::vector<Lambda*> Lambdas;

//------------------------------------------------------------------------------

/**
 * @brief A parameter of a @p Lambda function.
 *
 * A @p Param knows its @p lambda() it belongs to.
 */
class Param : public DefNode {
private:
    Param(size_t gid, Type type, Lambda* lambda, size_t index, const std::string& name)
        : DefNode(gid, Node_Param, type, 0, name)
        , lambda_(lambda)
        , index_(index)
    {}

public:
    class Peek {
    public:
        Peek() {}
        Peek(Def def, Lambda* from)
            : def_(def)
            , from_(from)
        {}

        Def def() const { return def_; }
        Lambda* from() const { return from_; }

    private:
        Def def_;
        Lambda* from_;
    };

    Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    std::vector<Peek> peek() const;

private:
    Lambda* const lambda_;
    const size_t index_;

    friend class World;
    friend class Lambda;
};

//------------------------------------------------------------------------------

enum class Intrinsic : uint8_t {
    None,                       ///< Not an intrinsic.
    _Accelerator_Begin = None,
    CUDA = _Accelerator_Begin,  ///< Internal CUDA-Backend.
    NVVM,                       ///< Internal NNVM-Backend.
    SPIR,                       ///< Internal SPIR-Backend.
    OpenCL,                     ///< Internal OpenCL-Backend.
    Parallel,                   ///< Internal Parallel-CPU-Backend.
    Spawn,                      ///< Internal Parallel-CPU-Backend.
    Sync,                       ///< Internal Parallel-CPU-Backend.
    Vectorize,                  ///< External vectorizer.
    _Accelerator_End,
    Mmap = _Accelerator_End,    ///< Intrinsic memory-mapping function.
    Munmap,                     ///< Intrinsic memory-unmapping function.
    Atomic,                     ///< Intrinsic atomic function
    Branch,                     ///< branch(cond, T, F).
    EndScope,                   ///< Dummy function which marks the end of a @p Scope.
};

enum class CC : uint8_t {
    C,          ///< C calling convention.
    Device,     ///< Device calling convention. These are special functions only available on a particular device.
};


/** 
 * @brief A function abstraction.
 *
 * A @p Lambda is always of function type @p FnTypeNode.
 * Each element of this function type is associated a properly typed @p Param - retrieved via @p params().
 */
class Lambda : public DefNode {
private:
    Lambda(size_t gid, FnType fn, CC cc, Intrinsic intrinsic, bool is_sealed, const std::string& name)
        : DefNode(gid, Node_Lambda, fn, 0, name)
        , parent_(this)
        , cc_(cc)
        , intrinsic_(intrinsic)
        , is_sealed_(is_sealed)
        , is_visited_(false)
    {
        params_.reserve(fn->num_args());
    }
    virtual ~Lambda() { for (auto param : params()) delete param; }

public:
    Lambda* stub() const { Type2Type map; return stub(map); }
    Lambda* stub(const std::string& name) const { Type2Type map; return stub(map, name); }
    Lambda* stub(Type2Type& type2type) const { return stub(type2type, name); }
    Lambda* stub(Type2Type& type2type, const std::string& name) const;
    Lambda* update_to(Def def) { return update_op(0, def); }
    Lambda* update_op(size_t i, Def def);
    Lambda* update_arg(size_t i, Def def) { return update_op(i+1, def); }
    const Param* append_param(Type type, const std::string& name = "");
    Lambdas direct_preds() const;
    Lambdas direct_succs() const;
    Lambdas indirect_preds() const;
    Lambdas indirect_succs() const;
    Lambdas preds() const;
    Lambdas succs() const;
    ArrayRef<const Param*> params() const { return params_; }
    Array<Def> params_as_defs() const;
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    const Param* mem_param() const;
    Def to() const;
    ArrayRef<Def> args() const { return empty() ? ArrayRef<Def>(0, 0) : ops().skip_front(); }
    Def arg(size_t i) const { return args()[i]; }
    FnType type() const { return DefNode::type().as<FnType>(); }
    FnType to_fn_type() const { return to()->type().as<FnType>(); }
    FnType arg_fn_type() const;
    size_t num_args() const { return args().size(); }
    size_t num_params() const { return params().size(); }
    Intrinsic& intrinsic() { return intrinsic_; }
    Intrinsic intrinsic() const { return intrinsic_; }
    CC& cc() { return cc_; }
    CC cc() const { return cc_; }
    void set_intrinsic(); ///< Sets @p intrinsic_ derived on this @p Lambda's @p name.
    bool is_external() const;
    void make_external();
    void make_internal();
    /**
     * Is this Lambda part of a call-lambda-cascade? <br>
     * @code
lambda(...) jump (foo, [..., lambda(...) ..., ...]
     * @endcode
     */
    bool is_cascading() const;
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const;
    bool is_accelerator() const;
    bool visit_capturing_intrinsics(std::function<bool(Lambda*)> func) const;
    bool is_passed_to_accelerator() const {
        return visit_capturing_intrinsics([&] (Lambda* lambda) { return lambda->is_accelerator(); });
    }
    bool is_passed_to_intrinsic(Intrinsic intrinsic) const {
        return visit_capturing_intrinsics([&] (Lambda* lambda) { return lambda->intrinsic() == intrinsic; });
    }
    void dump_head() const;
    void dump_jump() const;
    void destroy_body();
    void refresh();

    // terminate

    void jump(Def to, ArrayRef<Def> args);
    void jump(JumpTarget&);
    void branch(Def cond, Def t, Def f);
    std::pair<Lambda*, Def> call(Def to, ArrayRef<Def> args, Type ret_type);

    // value numbering

    Def set_value(size_t handle, Def def);
    Def get_value(size_t handle, Type type, const char* name = "");
    Def set_mem(Def def);
    Def get_mem();
    Lambda* parent() const { return parent_; }            ///< See @p parent_ for more information.
    void set_parent(Lambda* parent) { parent_ = parent; } ///< See @p parent_ for more information.
    void seal();
    bool is_sealed() const { return is_sealed_; }
    void unseal() { is_sealed_ = false; }
    void clear_value_numbering_table() { values_.clear(); }
    bool is_cleared() { return values_.empty(); }

private:
    class Todo {
    public:
        Todo() {}
        Todo(size_t handle, size_t index, Type type, const char* name)
            : handle_(handle)
            , index_(index)
            , type_(type)
            , name_(name)
        {}

        size_t handle() const { return handle_; }
        size_t index() const { return index_; }
        Type type() const { return type_; }
        const char* name() const { return name_; }

    private:
        size_t handle_;
        size_t index_;
        Type type_;
        const char* name_;
    };

    Def fix(size_t handle, size_t index, Type type, const char* name);
    Def try_remove_trivial_param(const Param*);
    Def find_def(size_t handle);
    void increase_values(size_t handle) { if (handle >= values_.size()) values_.resize(handle+1); }

    struct ScopeInfo {
        ScopeInfo(const Scope* scope)
            : scope(scope)
            , index(-1)
        {}

        const Scope* scope;
        size_t index;
    };

    std::list<ScopeInfo>::iterator list_iter(const Scope*);
    ScopeInfo* find_scope(const Scope*);
    ScopeInfo* register_scope(const Scope* scope) { scopes_.emplace_front(scope); return &scopes_.front(); }
    void unregister_scope(const Scope* scope) { scopes_.erase(list_iter(scope)); }

    /**
     * There exist three cases to distinguish here.
     * - @p parent_ == this: This @p Lambda is considered as a basic block, i.e.,
     *                       SSA construction will propagate value through this @p Lambda's predecessors.
     * - @p parent_ == nullptr: This @p Lambda is considered as top level function, i.e.,
     *                          SSA construction will stop propagate values here.
     *                          Any @p get_value which arrives here without finding a definition will return @p bottom.
     * - otherwise: This @p Lambda is considered as function head nested in @p parent_.
     *              Any @p get_value which arrives here without finding a definition will recursively try to find one in @p parent_.
     */
    Lambda* parent_;
    std::vector<const Param*> params_;
    std::list<ScopeInfo> scopes_;
    std::vector<Def> values_;
    std::vector<Todo> todos_;
    CC cc_;
    Intrinsic intrinsic_;
    mutable uint32_t reachable_ = 0;
    bool is_sealed_  : 1;
    bool is_visited_ : 1;

    friend class Cleaner;
    friend class Scope;
    friend class CFA;
    friend class World;
};

//------------------------------------------------------------------------------

template<class To>
using LambdaMap     = HashMap<Lambda*, To, GIDHash<Lambda*>, GIDEq<Lambda*>>;
using LambdaSet     = HashSet<Lambda*, GIDHash<Lambda*>, GIDEq<Lambda*>>;
using Lambda2Lambda = LambdaMap<Lambda*>;

//------------------------------------------------------------------------------

}

#endif
