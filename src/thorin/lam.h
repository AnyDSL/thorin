#ifndef THORIN_LAM_H
#define THORIN_LAM_H

#include "thorin/def.h"

namespace thorin {

/// A function type AKA Pi type.
class Pi : public Def {
protected:
    /// Constructor for a @em structural Pi.
    Pi(const Def* type, const Def* domain, const Def* codomain, const Def* dbg)
        : Def(Node, type, {domain, codomain}, 0, dbg)
    {}
    /// Constructor for a @em nominal Pi.
    Pi(const Def* type, const Def* dbg)
        : Def(Node, type, 2, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* domain() const { return op(0); }
    const Def* domain(size_t i) const { return proj(domain(), num_domains(), i); }
    Array<const Def*> domains() const { return Array<const Def*>(num_domains(), [&](size_t i) { return domain(i); }); }
    size_t num_domains() const { return domain()->num_outs(); }

    const Def* codomain() const { return op(1); }
    const Def* codomain(size_t i) const { return proj(codomain(), num_codomains(), i); }
    Array<const Def*> codomains() const { return Array<const Def*>(num_codomains(), [&](size_t i) { return codomain(i); }); }
    size_t num_codomains() const { return codomain()->num_outs(); }

    bool is_cn() const { return codomain()->isa<Bot>(); }
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;
    //@}
    /// @name setters for @em nominal @p Pi.
    //@{
    Pi* set_domain(const Def* domain) { return Def::set(0, domain)->as<Pi>(); }
    Pi* set_domain(Defs domains);
    Pi* set_codomain(const Def* codomain) { return Def::set(1, codomain)->as<Pi>(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Pi* stub(World&, const Def*, const Def*) override;
    const Pi* restructure();
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Pi;
    friend class World;
};

class Lam : public Def {
public:
    // TODO make these thigns axioms
    enum class Intrinsic : u8 {
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
        PeInfo,                     ///< Partial evaluation debug info.
    };

    /// calling convention
    enum class CC : u8 {
        C,          ///< C calling convention.
        Device,     ///< Device calling convention. These are special functions only available on a particular device.
    };

private:
    Lam(const Pi* pi, const Def* filter, const Def* body, const Def* dbg)
        : Def(Node, pi, {filter, body}, 0, dbg)
    {}
    Lam(const Pi* pi, CC cc, Intrinsic intrinsic, const Def* dbg)
        : Def(Node, pi, 2, u64(cc) << 8_u64 | u64(intrinsic), dbg)
    {}

public:
    /// @name type
    //@{
    const Pi* type() const { return Def::type()->as<Pi>(); }
    const Def* domain() const { return type()->domain(); }
    const Def* domain(size_t i) const { return type()->domain(i); }
    Array<const Def*> domains() const { return type()->domains(); }
    size_t num_domains() const { return type()->num_domains(); }
    const Def* codomain() const { return type()->codomain(); }
    //@}
    /// @name ops
    //@{
    const Def* filter() const { return op(0); }
    const Def* body() const { return op(1); }
    //@}
    /// @name params
    //@{
    const Def* mem_param(const Def* dbg = {});
    const Def* ret_param(const Def* dbg = {});
    //@}
    /// @name setters
    //@{
    Lam* set(size_t i, const Def* def) { return Def::set(i, def)->as<Lam>(); }
    Lam* set(Defs ops) { return Def::set(ops)->as<Lam>(); }
    Lam* set(const Def* filter, const Def* body) { return set({filter, body}); }
    Lam* set_filter(const Def* filter) { return set(0_s, filter); }
    Lam* set_body(const Def* body) { return set(1, body); }
    //@}
    /// @name setters: sets filter to @c false and sets the body by @p App -ing
    //@{
    void app(const Def* callee, const Def* arg, const Def* dbg = {});
    void app(const Def* callee, Defs args, const Def* dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, const Def* mem, const Def* dbg = {});
    void match(const Def* val, Defs cases, const Def* mem, const Def* dbg = {});
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Lam* stub(World&, const Def*, const Def*) override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}
    /// @name get/set fields - Intrinsic and CC
    //@{
    Intrinsic intrinsic() const { return Intrinsic(fields() & 0x00ff_u64); }
    void set_intrinsic(Intrinsic intrin) { fields_ = (fields_ & 0xff00_u64) | u64(intrin); }
    void set_intrinsic(); ///< Sets Intrinsic derived on this @p Lam's @p name.
    CC cc() const { return CC(fields() >> 8_u64); }
    void set_cc(CC cc) { fields_ = (fields_ & 0x00ff_u64) | u64(cc) << 8_u64; }
    //@}

    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const;
    bool is_accelerator() const;

    static constexpr auto Node = Node::Lam;
    friend class World;
};

template<class To>
using LamMap  = GIDMap<Lam*, To>;
using LamSet  = GIDSet<Lam*>;
using Lam2Lam = LamMap<Lam*>;

class App : public Def {
private:
    App(const Axiom* axiom, u16 currying_depth, const Def* type, const Def* callee, const Def* arg, const Def* dbg)
        : Def(Node, type, {callee, arg}, 0, dbg)
    {
        axiom_depth_.set(axiom, currying_depth);
    }

public:
    /// @name ops
    ///@{
    const Def* callee() const { return op(0); }
    const App* decurry() const { return callee()->as<App>(); } ///< Returns the @p callee again as @p App.
    const Pi* callee_type() const { return callee()->type()->as<Pi>(); }
    const Def* arg() const { return op(1); }
    const Def* arg(size_t i, const Def* dbg = {}) const { return arg()->out(i, dbg); }
    Array<const Def*> args() const { return arg()->outs(); }
    size_t num_args() const { return arg()->num_outs(); }
    //@}
    /// @name split arg
    //@{
    template<size_t N = size_t(-1), class F> auto args(F f) const { return arg()->split<N, F>(f); }
    template<size_t N = size_t(-1)> auto args() const { return arg()->split<N>(); }
    //@}
    /// @name get axiom and current currying depth
    //@{
    const Axiom* axiom() const { return axiom_depth_.ptr(); }
    u16 currying_depth() const { return axiom_depth_.index(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::App;
    friend class World;
};

inline Stream& operator<<(Stream& s, std::pair<Lam*, Lam*> p) { return operator<<(s, std::pair<const Def*, const Def*>(p.first, p.second)); }

// TODO remove - deprecated
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

// TODO remove - deprecated
size_t get_param_index(const Def* def);
Lam* get_param_lam(const Def* def);
std::vector<Peek> peek(const Def*);

}

#endif
