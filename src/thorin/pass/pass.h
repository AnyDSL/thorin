#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include <stack>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class PassMan;
typedef size_t undo_t;
static constexpr undo_t No_Undo = std::numeric_limits<undo_t>::max();

/// All Passes that want to be registered in the @p PassMan must implement this interface.
/// * Directly inherit from this class if your pass doesn't need state and a fixed-point iteration (a ReWrite pass).
/// * Inherit from @p FPPass using CRTP if you do need state.
class RWPassBase {
public:
    RWPassBase(PassMan& man, const std::string& name);
    virtual ~RWPassBase() {}

    /// @name getters
    //@{
    PassMan& man() { return man_; }
    const PassMan& man() const { return man_; }
    const std::string& name() const { return name_; }
    size_t proxy_id() const { return proxy_id_; }
    World& world();
    //@}

    /// @name hooks for the PassMan
    //@{
    virtual bool inspect() const = 0;

    /// Invoked just before @p rewrite%ing @p PassMan::curr_nom's body.
    virtual void enter() {}

    /// Rewrites a @em structural @p def within @p PassMan::curr_nom. Returns the replacement.
    virtual const Def* rewrite(const Def* def) { return def; }
    virtual const Def* rewrite(const Var* var) { return var; }
    virtual const Def* rewrite(const Proxy* proxy) { return proxy; }
    //@}

    /// @name Proxy
    //@{
    const Proxy* proxy(const Def* type, Defs ops, flags_t flags = 0, const Def* dbg = {}) { return world().proxy(type, ops, proxy_id(), flags, dbg); }
    /// @name Check whether given @p def is a Proxy whose index matches this @p Pass's @p index.
    const Proxy* isa_proxy(const Def* def, flags_t flags = 0) {
        if (auto proxy = def->isa<Proxy>(); proxy != nullptr && proxy->id() == proxy_id() && proxy->flags() == flags) return proxy;
        return nullptr;
    }
    const Proxy* as_proxy(const Def* def, flags_t flags = 0) {
        auto proxy = def->as<Proxy>();
        assert(proxy->id() == proxy_id() && proxy->flags() == flags);
        return proxy;
    }
    //@}

protected:
    template<class N> bool inspect() const;
    template<class N> N* curr_nom() const;

private:
    PassMan& man_;
    std::string name_;
    size_t proxy_id_;

    friend class PassMan;
};

/// Base class for all FPPass%es.
class FPPassBase : public RWPassBase {
public:
    FPPassBase(PassMan& man, const std::string& name);

    size_t index() const { return index_; }

    /// @name hooks for the PassMan
    //@{
    /// Invoked after the @p PassMan has @p finish%ed @p rewrite%ing @p curr_nom to analyze the @p Def.
    /// Return @p No_Undo or the state to roll back to.
    virtual undo_t analyze(const Def*  ) { return No_Undo; }
    virtual undo_t analyze(const Var*  ) { return No_Undo; }
    virtual undo_t analyze(const Proxy*) { return No_Undo; }
    virtual void* alloc() = 0;
    virtual void* copy(const void*) = 0;
    virtual void dealloc(void*) = 0;
    //@}

private:
    size_t index_;

    friend class PassMan;
};

/// An optimizer that combines several optimizations in an optimal way.
/// This is loosely based upon:
/// "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
class PassMan {
public:
    PassMan(World& world)
        : world_(world)
    {}

    /// @name getters
    //@{
    World& world() const { return world_; }
    const auto& passes() const { return passes_; }
    const auto& rw_passes() const { return rw_passes_; }
    const auto& fp_passes() const { return fp_passes_; }
    Def* curr_nom() const { return curr_nom_; }
    //@}

    /// @name create and run passes
    //@{
    /// Add a pass to this @p PassMan.
    template<class P, class... Args>
    P* add(Args&&... args) {
        auto p = std::make_unique<P>(*this, std::forward<Args>(args)...);
        passes_.emplace_back(p.get());
        auto res = p.get();

        if constexpr (std::is_base_of<FPPassBase, P>::value) {
            fp_passes_.emplace_back(std::move(p));
        } else {
            rw_passes_.emplace_back(std::move(p));
        }

        return res;
    }

    /// Run all registered passes on the whole @p world.
    void run();
    //@}

private:
    /// @name state
    //@{
    struct State {
        State() = default;
        State(const State&) = delete;
        State(State&&) = delete;
        State& operator=(State) = delete;
        State(size_t num)
            : data(num)
        {}

        Def* curr_nom = nullptr;
        DefArray old_ops;
        std::stack<Def*> stack;
        NomMap<undo_t> nom2visit;
        Array<void*> data;
        Def2Def old2new;
        DefSet analyzed;
    };

    void push_state();
    void pop_states(undo_t undo);
    State& curr_state() { assert(!states_.empty()); return states_.back(); }
    const State& curr_state() const { assert(!states_.empty()); return states_.back(); }
    undo_t curr_undo() const { return states_.size()-1; }
    //@}

    /// @name rewriting
    //@{
    const Def* rewrite(const Def*);

    const Def* map(const Def* old_def, const Def* new_def) {
        curr_state().old2new[old_def] = new_def;
        curr_state().old2new.emplace(new_def, new_def);
        return new_def;
    }

    std::optional<const Def*> lookup(const Def* old_def) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i)
            if (auto new_def = i->old2new.lookup(old_def)) return *new_def;
        return {};
    }
    //@}

    /// @name analyze
    //@{
    undo_t analyze(const Def*);
    bool analyzed(const Def* def) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            if (i->analyzed.contains(def)) return true;
        }
        curr_state().analyzed.emplace(def);
        return false;
    }
    //@}

    World& world_;
    std::vector<RWPassBase*> passes_;
    std::vector<std::unique_ptr<RWPassBase>> rw_passes_;
    std::vector<std::unique_ptr<FPPassBase>> fp_passes_;
    std::deque<State> states_;
    Def* curr_nom_ = nullptr;
    bool proxy_ = false;

    template<class P, class N> friend class FPPass;
};

template<class N = Def>
class RWPass : public RWPassBase {
public:
    RWPass(PassMan& man, const std::string& name)
        : RWPassBase(man, name)
    {}

protected:
    bool inspect() const override { return RWPassBase::inspect<N>(); }
    N* curr_nom() const { return RWPassBase::curr_nom<N>(); }
};

/// Inherit from this class using CRTP if you do need a Pass with a state.
template<class P, class N = Def>
class FPPass : public FPPassBase {
public:
    FPPass(PassMan& man, const std::string& name)
        : FPPassBase(man, name)
    {}

    /// @name memory management for state
    //@{
    void* alloc() override { return new typename P::Data(); }                                                     ///< Default ctor.
    void* copy(const void* p) override { return new typename P::Data(*static_cast<const typename P::Data*>(p)); } ///< Copy ctor.
    void dealloc(void* state) override { delete static_cast<typename P::Data*>(state); }                          ///< Dtor.
    //@}

protected:
    bool inspect() const override { return RWPassBase::inspect<N>(); }
    N* curr_nom() const { return RWPassBase::curr_nom<N>(); }

    /// @name state-related getters
    //@{
    auto& states() { return man().states_; }
    auto& data() { assert(!states().empty()); return *static_cast<typename P::Data*>(states().back().data[index()]); }
    /// Use this for your convenience if @c P::Data is a map.
    template<class K> auto& data(const K& key) { return data()[key]; }
    //@}

    /// @name undo getters
    //@{
    undo_t curr_undo() const { return man().curr_undo(); }

    undo_t undo_visit(Def* nom) const {
        if (auto undo = man().curr_state().nom2visit.lookup(nom)) return *undo;
        return No_Undo;
    }

    undo_t undo_enter(Def* nom) const {
        for (auto i = man().states_.size(); i-- != 0;)
            if (man().states_[i].curr_nom == nom) return i;
        return No_Undo;
    }
    //@}
};

inline World& RWPassBase::world() { return man().world(); }

template<class N>
bool RWPassBase::inspect() const {
    if constexpr(std::is_same<N, Def>::value)
        return man().curr_nom() ;
    else
        return man().curr_nom()->template isa<N>();
}

template<class N>
N* RWPassBase::curr_nom() const {
    if constexpr(std::is_same<N, Def>::value)
        return man().curr_nom() ;
    else
        return man().curr_nom()->template as<N>();
}

}

#endif
