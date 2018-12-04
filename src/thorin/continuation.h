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
    Param(const Type* type, Continuation* continuation, Debug dbg)
        : Def(Node_Param, type, 0, dbg)
        , continuation_(continuation)
    {}

public:
    Continuation* continuation() const { return continuation_; }

private:
    Continuation* const continuation_;

    friend class World;
    friend class Continuation;
};

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

size_t get_param_index(const Def* def);
Continuation* get_param_continuation(const Def* def);
std::vector<Peek> peek(const Def*);

//------------------------------------------------------------------------------

enum class Intrinsic : uint8_t {
    None,                       ///< Not an intrinsic.
    _Accelerator_Begin,
    CUDA = _Accelerator_Begin,  ///< Internal CUDA-Backend.
    NVVM,                       ///< Internal NNVM-Backend.
    OpenCL,                     ///< Internal OpenCL-Backend.
    AMDGPU,                     ///< Internal AMDGPU-Backend.
    HLS,                        ///< Internal HLS-Backend.
    Parallel,                   ///< Internal Parallel-CPU-Backend.
    Spawn,                      ///< Internal Parallel-CPU-Backend.
    Sync,                       ///< Internal Parallel-CPU-Backend.
    CreateGraph,                ///< Internal Flow-Graph-Backend.
    CreateTask,                 ///< Internal Flow-Graph-Backend.
    CreateEdge,                 ///< Internal Flow-Graph-Backend.
    ExecuteGraph,               ///< Internal Flow-Graph-Backend.
    Vectorize,                  ///< External vectorizer.
    _Accelerator_End,
    Reserve = _Accelerator_End, ///< Intrinsic memory reserve function
    Atomic,                     ///< Intrinsic atomic function
    CmpXchg,                    ///< Intrinsic cmpxchg function
    Undef,                      ///< Intrinsic undef function
    Branch,                     ///< branch(cond, T, F).
    Match,                      ///< match(val, otherwise, (case1, cont1), (case2, cont2), ...)
    PeInfo,                     ///< Partial evaluation debug info.
    EndScope,                   ///< Dummy function which marks the end of a @p Scope.
};

enum class CC : uint8_t {
    C,          ///< C calling convention.
    Device,     ///< Device calling convention. These are special functions only available on a particular device.
};


/**
 * A function abstraction.
 * A @p Continuation is always of function type @p FnTypeNode.
 */
class Continuation : public Def {
private:
    Continuation(const FnType* fn, CC cc, Intrinsic intrinsic, Debug dbg)
        : Def(Node_Continuation, fn, 0, dbg)
        , cc_(cc)
        , intrinsic_(intrinsic)
    {
        contains_continuation_ = true;
    }
    virtual ~Continuation() { delete param_; }

public:
    Continuation* stub() const;
    const Def* append_param(const Type* type, Debug dbg = {});
    Continuations preds() const;
    Continuations succs() const;
    const Param* param() const { return param_; }
    size_t num_params() const;
    const Def* param(size_t i) const;
    Array<const Def*> params() const;
    const Def* mem_param() const;
    const Def* ret_param() const;
    const Def* callee() const;
    const Def* arg() const { return op(1); }
    size_t num_args() const;
    Array<const Def*> args() const;
    const Def* arg(size_t i) const;
    Debug& jump_debug() const { return jump_debug_; }
    Location jump_location() const { return jump_debug(); }
    Symbol jump_name() const { return jump_debug().name(); }
    const FnType* type() const { return Def::type()->as<FnType>(); }
    const FnType* callee_fn_type() const { return callee()->type()->as<FnType>(); }
    const FnType* arg_fn_type() const;
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
    void destroy_body();

    std::ostream& stream_head(std::ostream&) const;
    std::ostream& stream_jump(std::ostream&) const;
    void dump_head() const;
    void dump_jump() const;

    // terminate

    void jump(const Def* callee, const Def* arg, Debug dbg = {});
    void jump(const Def* callee, Defs args, Debug dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    void match(const Def* val, Continuation* otherwise, Defs patterns, ArrayRef<Continuation*> continuations, Debug dbg = {});
    void verify() const {
#if THORIN_ENABLE_CHECKS
        if (!empty()) {
            auto c = callee_fn_type();
            auto a = arg_fn_type();
            assertf(c == a, "continuation '{}' calls '{}' of type '{}' but call has type '{}'\n", this, callee(), c, a);
        }
#endif
    }
    Continuation* update_op(size_t i, const Def* def);
    Continuation* update_callee(const Def* callee) { return update_op(0, callee); }
    Continuation* update_arg(const Def* arg) { return update_op(1, arg); }
    void set_filter(const Def* filter) { filter_ = filter; }
    void set_filter(Defs filter);
    void set_all_true_filter();
    void destroy_filter() { filter_ = nullptr; }
    const Def* filter() const { return filter_; }
    const Def* filter(size_t i) const;

private:
    mutable Debug jump_debug_;

    const Param* param_ = nullptr;
    const Def* filter_ = nullptr; // TODO make this an op
    CC cc_;
    Intrinsic intrinsic_;

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
    Call(const Def* callee, const Def* arg)
        : callee_(callee)
        , arg_(arg)
    {}
    Call(const Call& call)
        : callee_(call.callee())
        , arg_(call.arg())
        , hash_(call.hash_)
    {}

    const Def* callee() const { return callee_; }
    const Def*& callee() { return callee_; }
    const Def* arg() const { return arg_; }
    const Def* arg(size_t i) const;
    size_t num_args() const;
    Array<const Def*> args() const;

    uint64_t hash() const {
        if (hash_ == 0) {
            assert(callee_ && arg_);
            hash_ = hash_begin(callee());
            hash_ = hash_combine(hash_, arg()->gid());
        }
        return hash_;
    }

    bool operator==(const Call& other) const { return this->callee() == other.callee() && this->arg() == other.arg(); }
    Call& operator=(Call other) { swap(*this, other); return *this; }

    friend void swap(Call& call1, Call& call2) {
        using std::swap;
        swap(call1.callee_,  call2.callee_);
        swap(call1.arg_,     call2.arg_);
        swap(call1.hash_,    call2.hash_);
    }

private:
    const Def* callee_ = nullptr;
    const Def* arg_ = nullptr;
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
