#ifndef THORIN_CONTINUATION_H
#define THORIN_CONTINUATION_H

#include <list>
#include <vector>
#include <queue>

#include "thorin/config.h"
#include "thorin/primop.h"
#include "thorin/type.h"

namespace thorin {

class Lam;
class Scope;

typedef std::vector<Lam*> Lams;

//------------------------------------------------------------------------------

/**
 * A parameter of a @p Lam function.
 * A @p Param knows its @p lambda() it belongs to.
 */
class Param : public Def {
private:
    Param(const Type* type, Lam* continuation, size_t index, Debug dbg);

public:
    Lam* continuation() const { return op(0)->as_nom<Lam>(); }
    size_t index() const { return index_; }

private:
    const size_t index_;

    friend class World;
    friend class Lam;
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
    App(const Defs ops, Debug dbg);

public:
    const Def* callee() const { return op(0); }
    const Def* arg(size_t i) const { return op(1 + i); }
    size_t num_args() const { return num_ops() - 1; }
    const Defs args() const { return ops().skip_front(); }
    const Def* rebuild(World&, const Type*, Defs) const override;

    Lams using_continuations() const {
        std::vector<Lam*> conts;
        for (auto use : uses()) {
            if (auto cont = use->isa_nom<Lam>())
                conts.push_back(cont);
        }
        return conts;
    }

    void jump(const Def* callee, Defs args, Debug dbg = {});
    void verify() const;

    friend class World;
};

//------------------------------------------------------------------------------

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
 * A @p Lam is always of function type @p FnTypeNode.
 * Each element of this function type is associated a properly typed @p Param - retrieved via @p params().
 */
class Lam : public Def {
public:
    struct Attributes {
        Intrinsic intrinsic = Intrinsic::None;
        CC cc = CC::C;

        Attributes(Intrinsic intrinsic) : intrinsic(intrinsic) {}
        Attributes(CC cc = CC::C) : cc(cc) {}
    };

private:
    Lam(const FnType* fn, const Attributes& attributes, Debug dbg);
    virtual ~Lam() { for (auto param : params()) delete param; }

public:
    const FnType* type() const { return Def::type()->as<FnType>(); }

    Lam* stub() const;
    const Param* append_param(const Type* type, Debug dbg = {});
    Lams preds() const;
    Lams succs() const;
    ArrayRef<const Param*> params() const { return params_; }
    Array<const Def*> params_as_defs() const;
    const Param* param(size_t i) const { assert(i < num_params()); return params_[i]; }
    const Param* mem_param() const;
    const Param* ret_param() const;
    size_t num_params() const { return params().size(); }

    // TODO only used in parallel.cpp to create a dummy value, should be refactored in something cleaner
    const FnType* arg_fn_type() const;

    Attributes& attributes() { return attributes_; }
    const Attributes& attributes() const { return attributes_; }
    Intrinsic intrinsic() const { return attributes().intrinsic; }
    CC cc() const { return attributes().cc; }
    void set_intrinsic(); ///< Sets @p intrinsic_ derived on this @p Lam's @p name.
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const { return attributes().intrinsic != Intrinsic::None; }
    bool is_external() const;
    bool is_imported() const { return is_external() && !has_body(); }
    bool is_exported() const { return is_external() && has_body(); }
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

    /// Called to kill the lambda
    void destroy(const char*);

    void jump(const Def* callee, Defs args, Debug dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    void match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> continuations, Debug dbg = {});
    void verify() const;

    const Filter* filter() const { return op(1)->as<Filter>(); }
    void set_filter(const Filter* f) {
        unset_op(1);
        set_op(1, f);
    }
    void destroy_filter();
    const Filter* all_true_filter() const;

    /// Counts how many time that lambda is truly used, excluding its own Params and counting reused Apps multiple times
    /// We need to count re-used apps multiple times because this function is used to make inlining decisions.
    bool can_be_inlined() const {
        size_t used = 0;
        for (auto use : uses()) {
            if (auto app = use->isa<App>())
                used += app->num_uses();
            else if (!use->isa<Param>())
                used++;
        }
        return used < 2;
    }

    std::vector<const Param*> params_;
    Attributes attributes_;
    bool dead_ = false;

    friend class Cleaner;
    friend class Scope;
    friend class CFA;
    friend class World;
};

bool visit_uses(Lam*, std::function<bool(Lam*)>, bool include_globals);
bool visit_capturing_intrinsics(Lam*, std::function<bool(Lam*)>, bool include_globals = true);
bool is_passed_to_accelerator(Lam*, bool include_globals = true);
bool is_passed_to_intrinsic(Lam*, Intrinsic, bool include_globals = true);

void jump_to_dropped_call(Lam* continuation, Lam* dropped, const Defs call);

//------------------------------------------------------------------------------

template<class To>
using ParamMap    = GIDMap<const Param*, To>;
using ParamSet    = GIDSet<const Param*>;
using Param2Param = ParamMap<const Param*>;

template<class To>
using LamMap           = GIDMap<Lam*, To>;
using LamSet           = GIDSet<Lam*>;
using Lam2Lam          = LamMap<Lam*>;

//------------------------------------------------------------------------------

}

#endif
