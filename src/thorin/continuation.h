#ifndef THORIN_CONTINUATION_H
#define THORIN_CONTINUATION_H

#include <list>
#include <vector>
#include <queue>

#include "thorin/config.h"
#include "thorin/primop.h"
#include "thorin/type.h"

namespace thorin {

class Continuation;
class Rewriter;
class Scope;

typedef std::vector<Continuation*> Continuations;

//------------------------------------------------------------------------------

/**
 * A parameter of a @p Continuation function.
 * A @p Param knows its @p continuation() it belongs to.
 */
class Param : public Def {
private:
    Param(World&, const Type* type, const Continuation* continuation, size_t index, Debug dbg);

public:
    Continuation* continuation() const { return op(0)->as_nom<Continuation>(); }
    size_t index() const { return index_; }

    const Def* rebuild(World&, const Type*, Defs) const override;
    bool equal(const Def*) const override;
    hash_t vhash() const override;
private:
    const size_t index_;

    friend class World;
    friend class Continuation;
};

class Filter : public Def {
private:
    Filter(World& world, const Defs defs, Debug dbg);

public:
    size_t size() const { return num_ops(); }
    const Def* condition(size_t i) const { return op(i); }
    bool is_empty() const { return num_ops() == 0; }
    const Filter* cut(ArrayRef<size_t> indices) const;
    const Def* rebuild(World&, const Type*, Defs ) const override;

    friend class World;
};

class App : public Def {
private:
    App(World&, const Defs ops, Debug dbg);

public:
    const Def* callee() const { return op(0); }
    const Def* arg(size_t i) const { return op(1 + i); }
    size_t num_args() const { return num_ops() - 1; }
    const Defs args() const { return ops().skip_front(); }
    const Def* rebuild(World&, const Type*, Defs) const override;

    Continuations using_continuations() const {
        std::vector<Continuation*> conts;
        for (auto use : uses()) {
            if (auto cont = use->isa_nom<Continuation>())
                conts.push_back(cont);
        }
        return conts;
    }

    void jump(const Def* callee, Defs args, Debug dbg = {});
    bool verify() const;

    friend class World;
};

//------------------------------------------------------------------------------

enum class CC : uint8_t {
    Thorin,         ///< Standard calling convention for everything that solely lives inside thorin.
    C,              ///< C calling convention.
    Device,         ///< Device calling convention. These are special functions only available on a particular device.
};

enum class Intrinsic : uint8_t {
    None,
    AcceleratorBegin,
    CUDA = AcceleratorBegin,    ///< Internal CUDA-Backend.
    NVVM,                       ///< Internal NNVM-Backend.
    OpenCL,                     ///< Internal OpenCL-Backend.
    OpenCL_SPIRV,               ///< Internal OpenCL-Backend.
    AMDGPUHSA,                  ///< Internal AMDGPU-HSA-Backend.
    AMDGPUPAL,                  ///< Internal AMDGPU-PAL-Backend.
    ShadyCompute,               ///< Internal Shady Compute Backend.
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
    Branch,                     ///< branch(mem, cond, T, F).
    Match,                      ///< match(mem, val, otherwise, (case1, cont1), (case2, cont2), ...)
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
        CC cc = CC::Thorin;

        Attributes(Intrinsic intrinsic) : intrinsic(intrinsic) {}
        Attributes(CC cc = CC::Thorin) : cc(cc) {}
    };

private:
    Continuation(World&, const FnType* pi, const Attributes& attributes, Debug dbg);
    virtual ~Continuation() { for (auto param : params()) delete param; }

public:
    const FnType* type() const { return Def::type()->as<FnType>(); }

    Continuation* stub(Rewriter&, const Type*) const override;
    void rebuild_from(Rewriter&, const Def* old) override;
    const Param* append_param(const Type* type, Debug dbg = {});
    Continuations preds() const;
    Continuations succs() const;
    ArrayRef<const Param*> params() const { return params_; }
    Array<const Def*> params_as_defs() const;
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    const Param* mem_param() const;
    const Param* ret_param() const;
    size_t num_params() const { return params().size(); }

    Attributes& attributes() { return attributes_; }
    const Attributes& attributes() const { return attributes_; }
    Intrinsic intrinsic() const { return attributes().intrinsic; }
    CC cc() const { return attributes().cc; }
    void set_intrinsic(); ///< Sets @p intrinsic_ derived on this @p Continuation's @p name.
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const { return attributes().intrinsic != Intrinsic::None; }

    /// @name visibility
    ///@{
    /// |               | `!is_external()` | `is_external()`               |
    /// |---------------|------------------|-------------------------------|
    /// | `has_body()`  | regular function | that function `is_exported()` |
    /// | `!has_body()` | intrinsic        | that function `is_imported()` |
    bool is_external() const;
    bool is_imported() const { return is_external() && !has_body(); }
    bool is_exported() const { return is_external() && has_body(); }
    ///@}

    // TODO: probably should be moved to Attributes
    bool is_channel() const { return name().find("channel") != std::string::npos; }
    bool is_pipe() const { return name().find("pipe") != std::string::npos; }
    bool is_accelerator() const;

    const App* body() const { return op(0)->as<App>(); }
    bool has_body() const { return !op(0)->isa<Bottom>(); }
    void set_body(const App* app) {
        unset_op(0);
        set_op(0, app);
    }

    /// Called to kill the continuation
    void destroy(const char*);

    void jump(const Def* callee, Defs args, Debug dbg = {});
    void branch(const Def* mem, const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    void match(const Def* mem, const Def* val, Continuation* otherwise, Defs patterns, ArrayRef<Continuation*> continuations, Debug dbg = {});
    bool verify() const;

    const Filter* filter() const { return op(1)->as<Filter>(); }
    void set_filter(const Filter* f) {
        unset_op(1);
        set_op(1, f);
    }
    void destroy_filter();
    const Filter* all_true_filter() const;

    /// Counts how many time that continuation is truly used, excluding its own Params and counting reused Apps multiple times
    /// We need to count re-used apps multiple times because this function is used to make inlining decisions.
    bool can_be_inlined() const {
        size_t potentially_called = 0;
        for (auto use : uses()) {
            if (auto app = use->isa<App>())
                potentially_called += app->num_uses();
            else if (!use->isa<Param>())
                potentially_called++;

            if (potentially_called >= 2)
                return false;
        }
        return true;
    }
    bool never_called() const {
        for (auto use : uses()) {
            if (auto app = use->isa<App>()) {
                if (app->num_uses() != 0) {
                    return false;
                }
            } else if (!use->isa<Param>()) {
                return false;
            }
        }
        return true;
    }

    bool dead_ = false;
    std::vector<const Param*> params_;
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

void jump_to_dropped_call(Continuation* continuation, Continuation* dropped, const Defs call);

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
