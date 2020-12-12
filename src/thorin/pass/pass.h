#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include "thorin/world.h"

namespace thorin {

class PassMan;
typedef size_t undo_t;
static constexpr undo_t No_Undo = std::numeric_limits<undo_t>::max();

/// All Passes that want to be registered in the @p PassMan must implement this interface.
/// * Directly inherit from this class if your pass doesn't need state and a fixed-point iteration (a ReWrite pass).
/// * Inherit from @p FPPass using CRTP if you do need state.
class RWPass {
public:
    RWPass(PassMan& man, const std::string& name)
        : man_(man)
        , name_(name)
    {}
    virtual ~RWPass() {}

    /// @name getters
    //@{
    PassMan& man() { return man_; }
    const PassMan& man() const { return man_; }
    const std::string& name() const { return name_; }
    World& world();
    template<class T = Def> T* cur_nom() const;
    //@}

    /// @name hooks for the PassMan
    //@{
    /// Invoked just before @p rewrite%ing @p PassMan::cur_nom's body.
    virtual void enter() {}

    /// Rewrites a @p nom%inal within @p PassMan::cur_nom. Returns the replacement or @c nullptr if nothing has been done.
    virtual const Def* rewrite([[maybe_unused]] Def* nom, [[maybe_unused]] const Def* type, [[maybe_unused]] const Def* dbg) { return nullptr; }

    /// Rewrites a @em structural @p def within @p PassMan::cur_nom @em before it has been @p rebuild. Returns the replacement or @c nullptr if nothing has been done.
    virtual const Def* rewrite([[maybe_unused]] const Def* def, [[maybe_unused]] const Def* type, [[maybe_unused]] Defs, [[maybe_unused]] const Def* dbg) { return nullptr; }

    /// Rewrites a @em structural @p def within @p PassMan::cur_nom. Returns the replacement.
    virtual const Def* rewrite(const Def* def) { return def; }

    /// Invoked just after @p rewrite%ing and before @p analyze%ing @p PassMan::cur_nom's body.
    virtual void finish() {}
    //@}

private:
    PassMan& man_;
    std::string name_;
};

/// Base class for all FPPass%es.
class FPPassBase : public RWPass {
public:
    FPPassBase(PassMan& man, const std::string& name, size_t index)
        : RWPass(man, name)
        , index_(index)
    {}

    size_t index() const { return index_; }

    /// @name create Proxy
    //@{
    const Proxy* proxy(const Def* type, Defs ops, flags_t flags, const Def* dbg = {}) { return world().proxy(type, ops, index(), flags, dbg); }
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

    /// @name hooks for the PassMan
    //@{
    /// Invoked after the @p PassMan has @p finish%ed @p rewrite%ing @p cur_nom to analyze @p def.
    /// Default implementation invokes the other @p analyze method for all @p extended_ops of @p cur_nom.
    /// Return @p No_Undo or the state to roll back to.
    virtual undo_t analyze();
    virtual undo_t analyze([[maybe_unused]] const Def* def) { return No_Undo; }
    //@}

    /// @name mangage state - dummy implementations here
    //@{
    virtual void* alloc() { return nullptr; }
    virtual void dealloc(void*) {}
    //@}

private:
    size_t index_;
};

/// An optimizer that combines several optimizations in an optimal way.
/// This is loosely based upon:
/// "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
class PassMan {
public:
    PassMan(World& world)
        : world_(world)
    {}

    World& world() const { return world_; }

    /// @name create and run
    //@{
    /// Add a pass to this @p PassMan.
    template<class P, class... Args>
    PassMan& add(Args... args) {
        if constexpr (std::is_base_of<FPPassBase, P>::value) {
            auto p = std::make_unique<P>(*this, fp_passes_.size(), std::forward<Args>(args)...);
            passes_.emplace_back(p.get());
            fp_passes_.emplace_back(std::move(p));
        } else {
            auto p = std::make_unique<P>(*this, std::forward<Args>(args)...);
            passes_.emplace_back(p.get());
            rw_passes_.emplace_back(std::move(p));
        }
        return *this;
    }
    /// Run all registered passes on the whole @p world.
    void run();
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
    //@}

    template<class T = Def> T* cur_nom() const {
        if constexpr(std::is_same<T, Def>::value)
            return cur_nom_ ;
        else
            return cur_nom_ ? cur_nom_->template isa<T>() : nullptr;
    }

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
    };

    void init_state();
    void push_state();
    void pop_states(undo_t undo);
    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    const Def* rewrite(const Def*);
    void enqueue(const Def*);

    bool enqueued(const Def* def) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            if (i->enqueued.contains(def)) return true;
        }
        cur_state().enqueued.emplace(def);
        return false;
    }

    World& world_;
    std::vector<RWPass*> passes_;
    std::vector<std::unique_ptr<RWPass    >> rw_passes_;
    std::vector<std::unique_ptr<FPPassBase>> fp_passes_;
    std::deque<State> states_;
    Def* cur_nom_ = nullptr;

    template<class P> friend class FPPass;
};

/// Inherit from this class using CRTP if you do need a Pass with a state.
template<class P>
class FPPass : public FPPassBase {
public:
    FPPass(PassMan& man, const std::string& name, size_t index)
        : FPPassBase(man, name, index)
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
    template<class K, size_t I = 0>
    auto put(const K& key) {
        for (undo_t undo = states().size(); undo-- != 0;) {
            auto& set = std::get<I>(data(undo));
            if (auto i = set.find(key); i != set.end()) return std::tuple(undo, false);
        }

        auto [_, inserted] = std::get<I>(data()).emplace(key);
        assert(inserted);
        return std::tuple(cur_undo(), true);
    }

    /// Searches states from back to top in the map @p M for @p key and inserts @p init if nothing is found.
    /// @return A triple: <code> TODO </code>.
    template<class M>
    typename M::mapped_type* find(const typename M::key_type& key) {
        for (undo_t undo = states().size(); undo-- != 0;) {
            auto& map = std::get<M>(data(undo));
            if (auto i = map.find(key); i != map.end()) return &i->second;
        }

        return nullptr;
    }

    /// Searches states from back to top in the map @p M for @p key and inserts @p init if nothing is found.
    /// @return A triple: <code> [ref_to_mapped_val, undo, inserted] </code>.
    template<class M>
    std::tuple<typename M::mapped_type&, undo_t, bool> insert(const typename M::key_type& key, typename M::mapped_type&& init = {}) {
        for (undo_t undo = states().size(); undo-- != 0;) {
            auto& map = std::get<M>(data(undo));
            if (auto i = map.find(key); i != map.end()) return {i->second, undo, false};
        }

        auto [i, inserted] = std::get<M>(data()).emplace(key, std::move(init));
        assert(inserted);
        return {i->second, cur_undo(), true};
    }

    /// Use when implementing your own @p RWPass::analyze to remember whether you have already seen @p def.
    /// @return @c true if already analyzed, @c false if not - but subsequent invocations will then yield @c true.
    bool analyzed(const Def* def) {
        for (auto i = states().rbegin(), e = states().rend(); i != e; ++i) {
            if (i->analyzed[index()].contains(def)) return true;
        }
        states().back().analyzed[index()].emplace(def);
        return false;
    }
    //@}

    /// Use as guard within @p analyze to rule out common @p def%s one is usually not interested in and only considers @p T as @p PasMMan::cur_nom.
    template<class T = Def>
    T* descend(const Def* def) {
        auto cur_nom = man().template cur_nom<T>();
        if (cur_nom == nullptr || def->is_const() || def->isa_nominal() || def->isa<Param>() || analyzed(def)) return nullptr;
        if (auto proxy = def->isa<Proxy>(); proxy && proxy->index() != index()) return nullptr;
        return cur_nom;
    }

    undo_t cur_undo() const { return man().states_.size()-1; }

private:
    /// @name state-related getters
    //@{
    auto& states() { return man().states_; }
    auto& data(size_t i) { return *static_cast<typename P::Data*>(states()[i].data[index()]); }
    auto& data() { assert(!states().empty()); return *static_cast<typename P::Data*>(states().back().data[index()]); }
    //@}
};

inline World& RWPass::world() { return man().world(); }
inline const App* is_callee(const Def* def, size_t i) { return i == 0 ? def->isa<App>() : nullptr; }

template<class T = Def> T* RWPass::cur_nom() const { return man().template cur_nom<T>(); }

}

#endif
