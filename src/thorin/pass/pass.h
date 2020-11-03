#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include "thorin/world.h"

namespace thorin {

class PassMan;
typedef size_t undo_t;
static constexpr undo_t No_Undo = std::numeric_limits<undo_t>::max();

/// All Pass%es that want to be registered in the @p PassMan must implement this interface.
/// * Directly inherit from this class if your pass doesn't need state.
/// * Inherit from @p Pass using CRTP if you do need state.
class PassBase {
public:
    PassBase(PassMan& man, size_t index, const std::string& name)
        : man_(man)
        , index_(index)
        , name_(name)
    {}
    virtual ~PassBase() {}

    /// @name getters
    //@{
    PassMan& man() { return man_; }
    const PassMan& man() const { return man_; }
    size_t index() const { return index_; }
    const std::string& name() const { return name_; }
    World& world();
    //@}
    /// @name hooks for the PassMan
    //@{
    /// Invoked just before @p rewrite%ing @p cur_nom's body.
    virtual void enter([[maybe_unused]] Def* cur_nom) {}

    /// Rewrites a @em structural @p def within @p cur_nom. Returns the replacement.
    virtual const Def* rewrite(Def* cur_nom, const Def* def) = 0;

    /// Invoked just after @p rewrite%ing and before @p analyze%ing @p cur_nom's body.
    virtual void finish([[maybe_unused]] Def* cur_nom) {}

    /// Invoked after the @p PassMan has @p finish%ed @p rewrite%ing @p cur_nom to analyze @p def.
    /// Default implementation invokes the other @p analyze method for all @p extended_ops of @p cur_nom.
    /// Return @p No_Undo or the state to roll back to.
    virtual undo_t analyze(Def* cur_nom) {
        undo_t undo = No_Undo;
        for (auto op : cur_nom->extended_ops())
            undo = std::min(undo, analyze(cur_nom, op));
        return undo;
    }
    virtual undo_t analyze([[maybe_unused]] Def* cur_nom, [[maybe_unused]] const Def* def) { return No_Undo; }
    //@}
    /// @name create Proxy
    const Proxy* proxy(const Def* type, Defs ops, flags_t flags, Debug dbg = {}) { return world().proxy(type, ops, index(), flags, dbg); }
    const Proxy* proxy(const Def* type, Defs ops, Debug dbg = {}) { return proxy(type, ops, 0, dbg); }
    //@{
    /// @name check whether given @c def is a Proxy whose index matches this Pass's index
    const Proxy* isa_proxy(const Def* def, flags_t flags = 0) {
        if (auto proxy = def->isa<Proxy>(); proxy != nullptr && proxy->index() == index() && proxy->flags() == flags) return proxy;
        return nullptr;
    }
    const Proxy* as_proxy(const Def* def, flags_t flags = 0) {
        auto proxy = def->as<Proxy>();
        assert(proxy->index() == index() && proxy->flags() == flags);
        return proxy;
    }
    //@}
    /// @name mangage state - dummy implementations here
    //@{
    virtual void* alloc() { return nullptr; }
    virtual void dealloc(void*) {}
    //@}

private:
    PassMan& man_;
    size_t index_;
    std::string name_;
};

/// An optimizer that combines several optimizations in an optimal way.
/// This is loosely based upon:
/// "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
class PassMan {
public:
    typedef std::unique_ptr<PassBase> PassPtr;

    PassMan(World& world)
        : world_(world)
    {}

    /// @name create and run
    //@{
    /// Add @p Pass to this @p PassMan.
    template<class P, class... Args>
    PassMan& create(Args... args) {
        passes_.emplace_back(std::make_unique<P>(*this, passes_.size()), std::forward<Args>(args)...);
        return *this;
    }
    /// Run all registered @p Pass%es on the whole @p world.
    void run();
    //@}
    /// @name getters
    //@{
    World& world() const { return world_; }
    size_t num_passes() const { return passes_.size(); }
    //@}
    /// @name working with the rewrite-map
    //@{
    const Def* map(const Def* old_def, const Def* new_def) {
        cur_state().old2new[old_def] = new_def;
        cur_state().old2new.emplace(new_def, new_def);
        return new_def;
    }

    std::optional<const Def*> lookup(const Def* old_def) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            const auto& old2new = i->old2new;
            if (auto i = old2new.find(old_def); i != old2new.end()) return i->second;
        }

        return {};
    }

    bool is_tainted(Def* nom) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            if (i->tainted.contains(nom)) return true;
        }

        return false;
    }
    bool mark_tainted(Def* nom) { return cur_state().tainted.emplace(nom).second; }
    //@}

private:
    struct State {
        State() = default;
        State(const State&) = delete;
        State(State&&) = delete;
        State& operator=(State) = delete;
        State(size_t num)
            : data(num)
            , analyzed(num)
        {}

        Def* cur_nom = nullptr;
        Array<void*> data;
        Array<DefSet> analyzed;
        DefSet enqueued;
        Array<const Def*> old_ops;
        std::stack<Def*> stack;
        Def2Def old2new;
        NomSet tainted;
    };

    void push_state();
    void pop_states(undo_t undo);
    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    void enter(Def*);
    const Def* rewrite(Def*, const Def*);
    void enqueue(const Def*);

    bool enqueued(const Def* def) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            if (i->enqueued.contains(def)) return true;
        }
        cur_state().enqueued.emplace(def);
        return false;
    }

    World& world_;
    std::vector<PassPtr> passes_;
    std::deque<State> states_;

    template<class P> friend class Pass;
};

inline World& PassBase::world() { return man().world(); }
inline bool ignore(Lam* lam) { return lam == nullptr || lam->is_external() || lam->is_intrinsic() || !lam->is_set(); }

/// Inherit from this class using CRTP if you do need a Pass with a state.
template<class P>
class Pass : public PassBase {
public:
    Pass(PassMan& man, size_t index, const std::string& name)
        : PassBase(man, index, name)
    {}

    //@}
    /// @name alloc/dealloc state
    //@{
    void* alloc() override { return new typename P::Data(); }
    void dealloc(void* state) override { delete static_cast<typename P::Data*>(state); }
    //@}

protected:
    /// @name search in the state stack
    //@{
    /// Searches states from back to top in the set @p S for @p key and puts it into @p S if not found.
    /// @return A triple: <code> [undo, inserted] </code>.
    template<class S>
    auto put(const typename S::key_type& key) {
        for (undo_t undo = states().size(); undo-- != 0;) {
            auto& set = std::get<S>(data(undo));
            if (auto i = set.find(key); i != set.end()) return std::tuple(undo, false);
        }

        auto [_, inserted] = std::get<S>(data()).emplace(key);
        assert(inserted);
        return std::tuple(states().size()-1, true);
    }

    /// Searches states from back to top in the map @p M for @p key and inserts @p init if nothing is found.
    /// @return A triple: <code> [iterator, undo, inserted] </code>.
    template<class M>
    std::tuple<typename M::mapped_type&, undo_t, bool> insert(const typename M::key_type& key, typename M::mapped_type&& init = {}) {
        for (undo_t undo = states().size(); undo-- != 0;) {
            auto& map = std::get<M>(data(undo));
            if (auto i = map.find(key); i != map.end()) return {i->second, undo, false};
        }

        auto [i, inserted] = std::get<M>(data()).emplace(key, std::move(init));
        assert(inserted);
        return {i->second, states().size()-1, true};
    }

    /// Use when implementing your own @p PassBase::analyze to remember whether you have already seen @p def.
    /// @return @c true if already analyzed, @c false if not - but subsequent invocations will then yield @c true.
    bool analyzed(const Def* def) {
        for (auto i = states().rbegin(), e = states().rend(); i != e; ++i) {
            if (i->analyzed[index()].contains(def)) return true;
        }
        states().back().analyzed[index()].emplace(def);
        return false;
    }

    /// Use as guard within @p analyze to rule out common @p def%s one is usually not interested in and only considers @p T as @p nom&inal.
    template<class T = Lam>
    T* descend(Def* nom, const Def* def) {
        auto cur_nom = nom->template isa<T>();
        if (cur_nom == nullptr || def->is_const() || def->isa_nominal() || def->isa<Param>() || analyzed(def)) return nullptr;
        if (auto proxy = def->isa<Proxy>(); proxy && proxy->index() != index()) return nullptr;
        return cur_nom;
    }

private:
    /// @name state-related getters
    //@{
    auto& states() { return man().states_; }
    auto& data(size_t i) { return *static_cast<typename P::Data*>(states()[i].data[index()]); }
    auto& data() { assert(!states().empty()); return *static_cast<typename P::Data*>(states().back().data[index()]); }
    //@}
};

}

#endif
