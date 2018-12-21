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
class Param;

typedef std::vector<Continuation*> Continuations;

//------------------------------------------------------------------------------

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
    Continuation(const Pi* pi, CC cc, Intrinsic intrinsic, Debug dbg);

public:
    //@{ operands
    const Def* filter() const { return op(0); }
    const Def* filter(size_t i) const;
    const Def* callee() const { return op(1); }
    const Def* arg() const { return op(2); }
    const Def* arg(size_t i) const;
    Array<const Def*> args() const;
    size_t num_args() const;
    //@}

    //@{ params
    const Param* param(Debug dbg = {}) const;
    size_t num_params() const;
    const Def* param(size_t i, Debug dbg = {}) const;
    Array<const Def*> params() const;
    const Def* mem_param() const;
    const Def* ret_param() const;
    //@}

    //@{ setters
    void set_filter(const Def* filter) { update_op(0, filter); }
    void set_filter(Defs filter);
    void set_all_true_filter();
    void destroy_filter();
    //@}

    //@{ type
    const Pi* type() const { return Def::type()->as<Pi>(); }
    const Type* domain() const { return type()->domain(); }
    const Type* codomain() const { return type()->codomain(); }
    //@}

    Def* vstub(World&, const Type*) const override;
    const Def* vrebuild(World&, const Type*, Defs) const override { THORIN_UNREACHABLE; }

    Continuations preds() const;
    Continuations succs() const;
    bool is_empty() const;
    Debug& jump_debug() const { return jump_debug_; }
    Location jump_location() const { return jump_debug(); }
    Symbol jump_name() const { return jump_debug().name(); }
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
        //auto c = callee_fn_type();
        //auto a = arg_fn_type();
        //assertf(c == a, "continuation '{}' calls '{}' of type '{}' but call has type '{}'\n", this, callee(), c, a);
#endif
    }

    // TODO remove this once we have a proper App node
    void update_op(size_t i, const Def* def);
    void update_callee(const Def* callee) { update_op(0, callee); }
    void update_arg(const Def* arg) { update_op(1, arg); }

private:
    mutable Debug jump_debug_;

    CC cc_;
    Intrinsic intrinsic_;

    friend class Cleaner;
    friend class Scope;
    friend class CFA;
    friend class World;
};

/**
 * A parameter of a @p Continuation function.
 * A @p Param's op isits @p continuation() it belongs to.
 */
class Param : public Def {
private:
    Param(const Type* type, const Continuation* continuation, Debug dbg)
        : Def(Node_Param, type, Defs{continuation}, dbg)
    {
        assert(continuation->is_nominal());
    }

public:
    Continuation* continuation() const { return op(0)->as_continuation(); }
    const Def* vrebuild(World&, const Type*, Defs) const override;

    friend class World;
};

bool visit_uses(Continuation*, std::function<bool(Continuation*)>, bool include_globals);
bool visit_capturing_intrinsics(Continuation*, std::function<bool(Continuation*)>, bool include_globals = true);
bool is_passed_to_accelerator(Continuation*, bool include_globals = true);
bool is_passed_to_intrinsic(Continuation*, Intrinsic, bool include_globals = true);

class App : public Def {
private:
    App(const Type* type, const Def* callee, const Def* arg, Debug dbg)
        : Def(Node_App, type, {callee, arg}, dbg)
    {}

public:
    const Def* callee() const { return op(0); }
    const Def* arg() const { return op(1); }


    size_t num_args() const;
    const Def* arg(size_t i) const;
    Array<const Def*> args() const;

    const Def* vrebuild(World&, const Type*, Defs) const override;

    friend class World;
};

//void jump_to_dropped_call(Continuation* src, Continuation* dst, const Call& call);

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
