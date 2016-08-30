#ifndef THORIN_CONTINUATION_H
#define THORIN_CONTINUATION_H

#include <list>
#include <vector>
#include <queue>

#include "thorin/def.h"
#include "thorin/type.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class JumpTarget;
class Continuation;
class Scope;

typedef std::vector<Continuation*> Continuations;

//------------------------------------------------------------------------------

/**
 * A parameter of a @p Continuation function.
 * A @p Param knows its @p continuation() it belongs to.
 */
class Param : public Def {
private:
    Param(const Type* type, Continuation* continuation, size_t index, const Location& loc, const std::string& name)
        : Def(Node_Param, type, 0, loc, name)
        , continuation_(continuation)
        , index_(index)
    {}

public:
    class Peek {
    public:
        Peek() {}
        Peek(const Def* def, Continuation* from)
            : def_(def)
            , from_(from)
        {}

        const Def* def() const { return def_; }
        Continuation* from() const { return from_; }

    private:
        const Def* def_;
        Continuation* from_;
    };

    Continuation* continuation() const { return continuation_; }
    size_t index() const { return index_; }
    std::vector<Peek> peek() const;
    const Param* is_mem() const { return type()->isa<MemType>() ? this : nullptr; }

private:
    Continuation* const continuation_;
    const size_t index_;

    friend class World;
    friend class Continuation;
};

//------------------------------------------------------------------------------

enum class Intrinsic : uint8_t {
    None,                       ///< Not an intrinsic.
    _Accelerator_Begin,
    CUDA = _Accelerator_Begin,  ///< Internal CUDA-Backend.
    NVVM,                       ///< Internal NNVM-Backend.
    SPIR,                       ///< Internal SPIR-Backend.
    OpenCL,                     ///< Internal OpenCL-Backend.
    Parallel,                   ///< Internal Parallel-CPU-Backend.
    Spawn,                      ///< Internal Parallel-CPU-Backend.
    Sync,                       ///< Internal Parallel-CPU-Backend.
    Vectorize,                  ///< External vectorizer.
    _Accelerator_End,
    Reserve = _Accelerator_End, ///< Intrinsic memory reserve function
    Atomic,                     ///< Intrinsic atomic function
    CmpXchg,                    ///< Intrinsic cmpxchg function
    Branch,                     ///< branch(cond, T, F).
    EndScope,                   ///< Dummy function which marks the end of a @p Scope.
    Bitcast,                    ///< Intrinsic for reinterpretation of one type as another one
    Select,                     ///< Intrinsic vector 'select' function
    Sizeof,                     ///< Sizeof intrinsic
    Shuffle,                    ///< Intrinsic vector 'shuffle' function
};

enum class CC : uint8_t {
    C,          ///< C calling convention.
    Device,     ///< Device calling convention. These are special functions only available on a particular device.
};


/**
 * A function abstraction.
 * A @p Continuation is always of function type @p FnTypeNode.
 * Each element of this function type is associated a properly typed @p Param - retrieved via @p params().
 */
class Continuation : public Def {
private:
    Continuation(const FnType* fn, const Location& loc, CC cc, Intrinsic intrinsic, bool is_sealed, const std::string& name)
        : Def(Node_Continuation, fn, 0, loc, name)
        , parent_(this)
        , cc_(cc)
        , intrinsic_(intrinsic)
        , is_sealed_(is_sealed)
        , is_visited_(false)
    {
        params_.reserve(fn->num_args());
    }
    virtual ~Continuation() { for (auto param : params()) delete param; }

public:
    Continuation* stub() const { Type2Type map; return stub(map); }
    Continuation* stub(const std::string& name) const { Type2Type map; return stub(map, name); }
    Continuation* stub(Type2Type& type2type) const { return stub(type2type, name); }
    Continuation* stub(Type2Type& type2type, const std::string& name) const;
    Continuation* update_callee(const Def* def) { return update_op(0, def); }
    Continuation* update_op(size_t i, const Def* def);
    Continuation* update_arg(size_t i, const Def* def) { return update_op(i+1, def); }
    const Param* append_param(const Type* type, const std::string& name = "");
    Continuations direct_preds() const;
    Continuations direct_succs() const;
    Continuations indirect_preds() const;
    Continuations indirect_succs() const;
    Continuations preds() const;
    Continuations succs() const;
    ArrayRef<const TypeParam*> type_params() const { return type()->type_params(); }
    const TypeParam* type_param(size_t i) const { return type_params()[i]; }
    size_t num_type_params() const { return type_params().size(); }
    ArrayRef<const Param*> params() const { return params_; }
    Array<const Def*> params_as_defs() const;
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    const Param* mem_param() const;
    const Def* callee() const;
    Types type_args() const { return type_args_; }
    const Type* type_arg(size_t i) const { return type_args_[i]; }
    size_t num_type_args() const { return type_args_.size(); }
    Defs args() const { return empty() ? Defs(0, 0) : ops().skip_front(); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Location& jump_loc() const { return jump_loc_; }
    const FnType* type() const { return Def::type()->as<FnType>(); }
    const FnType* callee_fn_type() const { return callee()->type()->as<FnType>(); }
    const FnType* arg_fn_type() const;
    size_t num_args() const { return args().size(); }
    size_t num_params() const { return params().size(); }
    Intrinsic& intrinsic() { return intrinsic_; }
    Intrinsic intrinsic() const { return intrinsic_; }
    CC& cc() { return cc_; }
    CC cc() const { return cc_; }
    void set_intrinsic(); ///< Sets @p intrinsic_ derived on this @p Continuation's @p name.
    bool is_external() const;
    void make_external();
    void make_internal();
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const;
    bool is_accelerator() const;
    bool visit_capturing_intrinsics(std::function<bool(Continuation*)> func) const;
    bool is_passed_to_accelerator() const {
        return visit_capturing_intrinsics([&] (Continuation* continuation) { return continuation->is_accelerator(); });
    }
    bool is_passed_to_intrinsic(Intrinsic intrinsic) const {
        return visit_capturing_intrinsics([&] (Continuation* continuation) { return continuation->intrinsic() == intrinsic; });
    }
    void destroy_body();
    void refresh(Def2Def&);

    std::ostream& stream_head(std::ostream&) const;
    std::ostream& stream_jump(std::ostream&) const;
    void dump_head() const;
    void dump_jump() const;

    // terminate

    void jump(const Def* to, Array<const Type*> type_args, Defs args, const Location& loc);
    void jump(JumpTarget&, const Location& loc);
    void branch(const Def* cond, const Def* t, const Def* f, const Location& loc);
    std::pair<Continuation*, const Def*> call(const Def* to, Types type_args, Defs args, const Type* ret_type, const Location& loc);

    // value numbering

    const Def* set_value(size_t handle, const Def* def);
    const Def* get_value(size_t handle, const Type* type, const char* name = "");
    const Def* set_mem(const Def* def);
    const Def* get_mem();
    Continuation* parent() const { return parent_; }            ///< See @p parent_ for more information.
    void set_parent(Continuation* parent) { parent_ = parent; } ///< See @p parent_ for more information.
    void seal();
    bool is_sealed() const { return is_sealed_; }
    void unseal() { is_sealed_ = false; }
    void clear_value_numbering_table() { values_.clear(); }
    bool is_cleared() { return values_.empty(); }

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

    const Def* fix(size_t handle, size_t index, const Type* type, const char* name);
    const Def* try_remove_trivial_param(const Param*);
    const Def* find_def(size_t handle);
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
    Array<const Type*> type_args_;
    Location jump_loc_;

    /**
     * There exist three cases to distinguish here.
     * - @p parent_ == this: This @p Continuation is considered as a basic block, i.e.,
     *                       SSA construction will propagate value through this @p Continuation's predecessors.
     * - @p parent_ == nullptr: This @p Continuation is considered as top level function, i.e.,
     *                          SSA construction will stop propagate values here.
     *                          Any @p get_value which arrives here without finding a definition will return @p bottom.
     * - otherwise: This @p Continuation is considered as function head nested in @p parent_.
     *              Any @p get_value which arrives here without finding a definition will recursively try to find one in @p parent_.
     */
    Continuation* parent_;
    std::vector<const Param*> params_;
    std::list<ScopeInfo> scopes_;
    std::deque<Tracker> values_;
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

struct Call {
    Call(Types type_args, Array<const Def*> ops)
        : type_args_(type_args)
        , ops_(ops)
    {}
    Call(Array<const Type*>&& type_args, Array<const Def*>&& ops)
        : type_args_(std::move(type_args))
        , ops_(std::move(ops))
    {}
    Call(const Call& call)
        : type_args_(call.type_args())
        , ops_(call.ops())
    {}
    Call(Call&& call)
        : type_args_(std::move(call.type_args_))
        , ops_(std::move(call.ops_))
    {}
    Call(const Continuation* continuation)
        : type_args_(continuation->num_type_args())
        , ops_(continuation->size())
    {}

    Types type_args() const { return type_args_; }
    size_t num_type_args() const { return type_args().size(); }
    const Type* type_arg(size_t i) const { return type_args_[i]; }
    const Type*& type_arg(size_t i) { return type_args_[i]; }

    Defs ops() const { return ops_; }
    size_t num_ops() const { return ops().size(); }
    const Def* op(size_t i) const { return ops_[i]; }
    const Def*& callee(size_t i) { return ops_[i]; }
    const Def* callee() const { return ops_.front(); }
    const Def*& callee() { return ops_.front(); }

    Defs args() const { return ops_.skip_front(); }
    size_t num_args() const { return args().size(); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Def*& arg(size_t i) { return ops_[i+1]; }

    bool operator==(const Call& other) const { return this->type_args() == other.type_args() && this->ops() == other.ops(); }
    Call& operator=(Call other) { swap(*this, other); return *this; }

    friend void swap(Call& call1, Call& call2) {
        using std::swap;
        swap(call1.type_args_, call2.type_args_);
        swap(call1.ops_,       call2.ops_);
    }

private:
    Array<const Type*> type_args_;
    Array<const Def*> ops_;
};

template<>
struct Hash<Call> {
    uint64_t operator () (const Call& call) const {
        uint64_t seed = hash_begin();
        for (auto type : call.type_args())
            seed = hash_combine(seed, type ? type->gid() : 0);
        for (auto arg : call.ops())
            seed = hash_combine(seed,  arg ?  arg->gid() : 0);
        return seed;
    }
};

void jump_to_cached_call(Continuation* src, Continuation* dst, const Call& call);

void clear_value_numbering_table(World&);

//------------------------------------------------------------------------------

template<class To>
using ParamMap     = HashMap<const Param*, To, GIDHash<const Param*>>;
using ParamSet     = HashSet<const Param*, GIDHash<const Param*>>;
using Param2Param  = ParamMap<const Param*>;

template<class To>
using ContinuationMap           = HashMap<Continuation*, To, GIDHash<Continuation*>>;
using ContinuationSet           = HashSet<Continuation*, GIDHash<Continuation*>>;
using Continuation2Continuation = ContinuationMap<Continuation*>;

//------------------------------------------------------------------------------

}

#endif
