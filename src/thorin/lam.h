#ifndef THORIN_LAM_H
#define THORIN_LAM_H

#include "thorin/def.h"

namespace thorin {

/// A function type AKA Pi type.
class Pi : public Def {
protected:
    /// Constructor for a @em structural Pi.
    Pi(const Def* type, const Def* dom, const Def* codom, const Def* dbg)
        : Def(Node, type, {dom, codom}, 0, dbg)
    {}
    /// Constructor for a @em nom Pi.
    Pi(const Def* type, const Def* dbg)
        : Def(Node, type, 2, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* dom() const { return op(0); }
    const Def* dom(size_t i) const { return proj(dom(), num_doms(), i); }
    DefArray doms() const { return DefArray(num_doms(), [&](size_t i) { return dom(i); }); }
    size_t num_doms() const { return dom()->num_outs(); }

    const Def* codom() const { return op(1); }
    const Def* codom(size_t i) const { return proj(codom(), num_codoms(), i); }
    DefArray codoms() const { return DefArray(num_codoms(), [&](size_t i) { return codom(i); }); }
    size_t num_codoms() const { return codom()->num_outs(); }

    bool is_cn() const;
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;
    //@}
    /// @name setters for @em nom @p Pi.
    //@{
    Pi* set_dom(const Def* dom) { return Def::set(0, dom)->as<Pi>(); }
    Pi* set_dom(Defs doms);
    Pi* set_codom(const Def* codom) { return Def::set(1, codom)->as<Pi>(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Pi* stub(World&, const Def*, const Def*) override;
    const Pi* restructure() override;
    //@}

    static constexpr auto Node = Node::Pi;
    friend class World;
};

class Lam : public Def {
public:
    /// calling convention
    enum class CC : u8 {
        C,          ///< C calling convention.
        Device,     ///< Device calling convention. These are special functions only available on a particular device.
    };

private:
    Lam(const Pi* pi, const Def* filter, const Def* body, const Def* dbg)
        : Def(Node, pi, {filter, body}, 0, dbg)
    {}
    Lam(const Pi* pi, CC cc, const Def* dbg)
        : Def(Node, pi, 2, u64(cc), dbg)
    {}

public:
    /// @name type
    //@{
    const Pi* type() const { return Def::type()->as<Pi>(); }
    const Def* dom() const { return type()->dom(); }
    const Def* dom(size_t i) const { return type()->dom(i); }
    DefArray doms() const { return type()->doms(); }
    size_t num_doms() const { return type()->num_doms(); }
    const Def* codom() const { return type()->codom(); }
    //@}
    /// @name ops
    //@{
    const Def* filter() const { return op(0); }
    const Def* body() const { return op(1); }
    //@}
    /// @name vars
    //@{
    const Def* mem_var(const Def* dbg = {});
    const Def* ret_var(const Def* dbg = {});
    //@}
    /// @name setters
    //@{
    Lam* set(size_t i, const Def* def) { return Def::set(i, def)->as<Lam>(); }
    Lam* set(Defs ops) { return Def::set(ops)->as<Lam>(); }
    Lam* set(const Def* filter, const Def* body) { return set({filter, body}); }
    Lam* set_filter(const Def* filter) { return set(0_s, filter); }
    Lam* set_filter(bool filter);
    Lam* set_body(const Def* body) { return set(1, body); }
    //@}
    /// @name setters: sets filter to @c false and sets the body by @p App -ing
    //@{
    void app(const Def* callee, const Def* arg, const Def* dbg = {});
    void app(const Def* callee, Defs args, const Def* dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, const Def* mem, const Def* dbg = {});
    void test(const Def* value, const Def* index, const Def* match, const Def* clash, const Def* mem, const Def* dbg = {});
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Lam* stub(World&, const Def*, const Def*) override;
    //@}
    /// @name get/set fields - CC
    //@{
    CC cc() const { return CC(fields()); }
    void set_cc(CC cc) { fields_ = u64(cc); }
    //@}

    bool is_basicblock() const;
    bool is_returning() const;

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
    DefArray args() const { return arg()->outs(); }
    size_t num_args() const { return arg()->num_outs(); }
    //@}

    /// @name split arg
    //@{
    template<size_t A = -1_s, class F = std::identity> auto args(          F f = {}) const { return arg()->outs<A, F>(   f); }
    template<                 class F = std::identity> auto args(size_t a, F f = {}) const { return arg()->outs<   F>(a, f); }
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

inline const App* isa_callee(const Def* def, size_t i) { return i == 0 ? def->isa<App>() : nullptr; }

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
size_t get_var_index(const Def* def);
Lam* get_var_lam(const Def* def);
std::vector<Peek> peek(const Def*);

inline bool ignore(Lam* lam) { return lam == nullptr || lam->is_external() || !lam->is_set(); }

}

#endif
