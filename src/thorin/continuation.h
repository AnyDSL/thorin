#ifndef THORIN_CONTINUATION_H
#define THORIN_CONTINUATION_H

#include <list>
#include <vector>
#include <queue>

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/type.h"

namespace thorin {

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
    Param(const Type* type, Continuation* continuation, size_t index, Debug dbg)
        : Def(Node_Param, type, 0, dbg)
        , continuation_(continuation)
        , index_(index)
    {}

public:
    Continuation* continuation() const { return continuation_; }
    size_t index() const { return index_; }

private:
    Continuation* const continuation_;
    const size_t index_;

    friend class World;
    friend class Continuation;
};

//------------------------------------------------------------------------------

enum class Visibility : uint8_t {
    Internal,   ///< Internal to the module (only visible from inside it)
    External    ///< External to the module (either imported or exported)
};

enum class CC : uint8_t {
    C,          ///< C calling convention.
    Device,     ///< Device calling convention. These are special functions only available on a particular device.
};

enum class Intrinsic : uint8_t {
    None,
    AcceleratorBegin,
    CUDA = AcceleratorBegin,    ///< Internal CUDA-Backend.
    NVVM,                       ///< Internal NNVM-Backend.
    OpenCL,                     ///< Internal OpenCL-Backend.
    AMDGPU,                     ///< Internal AMDGPU-Backend.
    HLS,                        ///< Internal HLS-Backend.
    Parallel,                   ///< Internal Parallel-CPU-Backend.
    Fibers,                     ///< Internal Parallel-CPU-Backend using resumable fibers.
    Spawn,                      ///< Internal Parallel-CPU-Backend.
    Sync,                       ///< Internal Parallel-CPU-Backend.
    Vectorize,                  ///< External vectorizer.
    AcceleratorEnd,
    Reserve = AcceleratorEnd,   ///< Intrinsic memory reserve function
    Atomic,                     ///< Intrinsic atomic function
    AtomicLoad,                 ///< Intrinsic atomic load function
    AtomicStore,                ///< Intrinsic atomic store function
    CmpXchg,                    ///< Intrinsic cmpxchg function
    CmpXchgWeak,                ///< Intrinsic cmpxchg weak function
    Fence,                      ///< Intrinsic fence function
    Undef,                      ///< Intrinsic undef function
    PipelineContinue,           ///< Intrinsic loop-pipelining-HLS-Backend
    Pipeline,                   ///< Intrinsic loop-pipelining-HLS-Backend
    Branch,                     ///< branch(cond, T, F).
    Match,                      ///< match(val, otherwise, (case1, cont1), (case2, cont2), ...)
    PeInfo,                     ///< Partial evaluation debug info.
    EndScope                    ///< Dummy function which marks the end of a @p Scope.
};

/**
 * A function abstraction.
 * A @p Continuation is always of function type @p FnTypeNode.
 * Each element of this function type is associated a properly typed @p Param - retrieved via @p params().
 */
class Continuation : public Def {
public:
    struct Attributes {
        Intrinsic intrinsic = Intrinsic::None;
        Visibility visibility = Visibility::Internal;
        CC cc = CC::C;

        Attributes() = default;
        Attributes(Intrinsic intrinsic) : intrinsic(intrinsic) {}
        Attributes(Visibility visibility, CC cc = CC::C) : visibility(visibility), cc(cc) {}
    };

private:
    Continuation(const FnType* fn, const Attributes& attributes, Debug dbg)
        : Def(Node_Continuation, fn, 0, dbg)
        , attributes_(attributes)
    {
        params_.reserve(fn->num_ops());
    }
    virtual ~Continuation() { for (auto param : params()) delete param; }

public:
    Continuation* stub() const;
    const Param* append_param(const Type* type, Debug dbg = {});
    Continuations preds() const;
    Continuations succs() const;
    ArrayRef<const Param*> params() const { return params_; }
    Array<const Def*> params_as_defs() const;
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    const Param* mem_param() const;
    const Param* ret_param() const;
    const Def* callee() const;
    Defs args() const { return num_ops() == 0 ? Defs(0, 0) : ops().skip_front(); }
    const Def* arg(size_t i) const { return args()[i]; }
    const FnType* type() const { return Def::type()->as<FnType>(); }
    const FnType* callee_fn_type() const { return callee()->type()->as<FnType>(); }
    const FnType* arg_fn_type() const;
    size_t num_args() const { return args().size(); }
    size_t num_params() const { return params().size(); }
    Attributes& attributes() { return attributes_; }
    const Attributes& attributes() const { return attributes_; }
    Intrinsic intrinsic() const { return attributes().intrinsic; }
    CC cc() const { return attributes().cc; }
    void set_intrinsic(); ///< Sets @p intrinsic_ derived on this @p Continuation's @p name.
    void make_external() { attributes().visibility = Visibility::External; }
    void make_internal() { attributes().visibility = Visibility::Internal; }
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const { return attributes().intrinsic != Intrinsic::None; }
    bool is_external() const { return attributes().visibility == Visibility::External; }
    bool is_internal() const { return attributes().visibility == Visibility::Internal; }
    bool is_imported() const { return is_external() && empty(); }
    bool is_exported() const { return is_external() && !empty(); }
    // TODO: probably should be moved to Attributes
    bool is_channel() const { return name().find("channel") != std::string::npos; }
    bool is_pipe() const { return name().find("pipe") != std::string::npos; }
    bool is_accelerator() const;
    void destroy_body();

    // terminate

    void jump(const Def* callee, Defs args, Debug dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    void match(const Def* val, Continuation* otherwise, Defs patterns, ArrayRef<Continuation*> continuations, Debug dbg = {});
    void verify() const {
#if THORIN_ENABLE_CHECKS
        auto c = callee_fn_type();
        auto a = arg_fn_type();
        assertf(c == a, "continuation '{}' calls '{}' of type '{}' but call has type '{}'\n", this, callee(), c, a);
#endif
    }
    Continuation* update_op(size_t i, const Def* def);
    Continuation* update_callee(const Def* def) { return update_op(0, def); }
    Continuation* update_arg(size_t i, const Def* def) { return update_op(i+1, def); }
    void set_filter(Defs defs) {
        assertf(defs.empty() || num_params() == defs.size(), "expected {} - got {}", num_params(), defs.size());
        filter_ = defs;
    }
    void set_all_true_filter();
    void destroy_filter() { filter_.shrink(0); }
    Defs filter() const { return filter_; }
    const Def* filter(size_t i) const { return filter_[i]; }

    std::vector<const Param*> params_;
    Array<const Def*> filter_; ///< used during @p partial_evaluation
    Attributes attributes_;

    friend class Cleaner;
    friend class Scope;
    friend class CFA;
    friend class World;
};

bool visit_uses(Continuation*, std::function<bool(Continuation*)>, bool include_globals);
bool visit_capturing_intrinsics(Continuation*, std::function<bool(Continuation*)>, bool include_globals = true);
bool is_passed_to_accelerator(Continuation*, bool include_globals = true);
bool is_passed_to_intrinsic(Continuation*, Intrinsic, bool include_globals = true);

struct Call {
    struct Hash {
        static uint64_t hash(const Call& call) { return call.hash(); }
        static bool eq(const Call& c1, const Call& c2) { return c1 == c2; }
        static Call sentinel() { return Call(); }
    };

    Call() {}
    Call(Array<const Def*> ops)
        : ops_(ops)
    {}
    Call(Array<const Def*>&& ops)
        : ops_(std::move(ops))
    {}
    Call(const Call& call)
        : ops_(call.ops())
        , hash_(call.hash_)
    {}
    Call(Call&& call)
        : ops_(std::move(call.ops_))
        , hash_(call.hash_)
    {}
    Call(size_t num_ops)
        : ops_(num_ops)
    {}

    Defs ops() const { return ops_; }
    size_t num_ops() const { return ops().size(); }
    const Def* op(size_t i) const { return ops_[i]; }
    const Def* callee() const { return ops_.front(); }
    const Def*& callee() { return ops_.front(); }

    Defs args() const { return ops_.skip_front(); }
    size_t num_args() const { return args().size(); }
    const Def* arg(size_t i) const { return args()[i]; }
    const Def*& arg(size_t i) { return ops_[i+1]; }

    uint64_t hash() const {
        if (hash_ == 0) {
            hash_ = hash_begin();
            for (auto op : ops())
                hash_ = hash_combine(hash_, op ? op->gid() : 0);
        }

        return hash_;
    }

    bool operator==(const Call& other) const { return this->ops() == other.ops(); }
    Call& operator=(Call other) { swap(*this, other); return *this; }
    explicit operator bool() { return !ops_.empty(); }

    friend void swap(Call& call1, Call& call2) {
        using std::swap;
        swap(call1.ops_,  call2.ops_);
        swap(call1.hash_, call2.hash_);
    }

private:
    Array<const Def*> ops_;
    mutable uint64_t hash_ = 0;
};

void jump_to_dropped_call(Continuation* src, Continuation* dst, const Call& call);

//------------------------------------------------------------------------------

template<class To>
using ParamMap    = GIDMap<const Param*, To>;
using ParamSet    = GIDSet<const Param*>;
using Param2Param = ParamMap<const Param*>;

template<class To>
using ContinuationMap           = GIDMap<Continuation*, To>;
using ContinuationSet           = GIDSet<Continuation*>;
using Continuation2Continuation = ContinuationMap<Continuation*>;

//------------------------------------------------------------------------------

}

#endif
