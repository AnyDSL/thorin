#if 0
#ifndef THORIN_lam_H
#define THORIN_lam_H

#include <list>
#include <vector>
#include <queue>

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/type.h"

namespace thorin {

class Lam;
class Param;

typedef std::vector<Lam*> Lams;

//------------------------------------------------------------------------------

class Peek {
public:
    Peek() {}
    Peek(const Def* def, Lam* from)
        : def_(def)
        , from_(from)
    {}

    const Def* def() const { return def_; }
    Lam* from() const { return from_; }

private:
    const Def* def_;
    Lam* from_;
};

size_t get_param_index(const Def* def);
Lam* get_param_lam(const Def* def);
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

bool visit_uses(Lam*, std::function<bool(Lam*)>, bool include_globals);
bool visit_capturing_intrinsics(Lam*, std::function<bool(Lam*)>, bool include_globals = true);
bool is_passed_to_accelerator(Lam*, bool include_globals = true);
bool is_passed_to_intrinsic(Lam*, Intrinsic, bool include_globals = true);

void app_to_dropped_app(Lam* src, Lam* dst, const App*);

//------------------------------------------------------------------------------

template<class To>
using AppMap  = GIDMap<const App*, To>;
using AppSet  = GIDSet<const App*>;
using App2App = AppMap<const App*>;

template<class To>
using ParamMap    = GIDMap<const Param*, To>;
using ParamSet    = GIDSet<const Param*>;
using Param2Param = ParamMap<const Param*>;

template<class To>
using LamMap  = GIDMap<Lam*, To>;
using LamSet  = GIDSet<Lam*>;
using Lam2Lam = LamMap<Lam*>;

//------------------------------------------------------------------------------

}

#endif
#endif
