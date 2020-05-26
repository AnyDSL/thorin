#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include <map>

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
    ///@}
    /// @name hooks for the PassMan
    //@{
    /// Inspects a @p nom%inal when first visited during @p rewrite%ing @p cur_nom.
    virtual void visit([[maybe_unused]] Def* cur_nom, [[maybe_unused]] Def* nom) {}

    /// Rewrites a @em structural @p def within @p cur_nom. Returns the replacement.
    virtual const Def* rewrite(Def* cur_nom, const Def* def) = 0;

    /// Invoked after the @p PassMan has finished @p rewrite%ing @p cur_nom to analyze @p def.
    /// Return @p No_Undo or the state to roll back to.
    virtual undo_t analyze([[maybe_unused]] Def* cur_nom, [[maybe_unused]] const Def* def) { return No_Undo; }
    ///@}
    /// @name Proxy-related operations
    const Proxy* proxy(const Def* type, Defs ops, Debug dbg = {}) { return world().proxy(type, ops, index(), dbg); }
    const Proxy* isa_proxy(const Def* def) { return isa<Proxy>(index(), def); }
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
    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { cur_state().old2new[old_def] = new_def; return new_def; }
    const Def* lookup(const Def* old_def) {
        if (auto new_def = _lookup(old_def)) {
            while (true) {
                world().DLOG("{} -> {}", old_def, new_def);
                old_def = new_def;
                new_def = _lookup(old_def);
                if (new_def == nullptr) return old_def;
                if (new_def == old_def) return new_def;
            }
        }

        return nullptr;
    }

    const Def* _lookup(const Def* old_def) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            const auto& old2new = i->old2new;
            if (auto i = old2new.find(old_def); i != old2new.end()) return i->second;
        }

        return nullptr;
    }
    bool is_tainted(Def* nom) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            if (i->tainted.contains(nom)) return true;
        }

        return false;
    }
    bool mark_tainted(Def* nom) { return cur_state().tainted.emplace(nom).second; }
    template<class T = Def> T*& reincarnate(T* old_nom) {
        auto [i, inserted] = reincanate_.emplace(old_nom, nullptr);
        assert(inserted || i->second->template isa<T>());
        return (T*&) i->second;
    }
    //@}

private:
    struct State {
        State() = default;
        State(const State&) = delete;
        State(State&&) = delete;
        State& operator=(State) = delete;
        State(size_t num)
            : data(num)
        {}

        Def* cur_nom = nullptr;
        Array<void*> data;
        Array<const Def*> old_ops;
        std::stack<Def*> stack;
        Def2Def old2new;
        DefSet analyzed;
        NomSet tainted;
    };

    void push_state();
    void pop_states(undo_t undo);
    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    void enter(Def*);
    const Def* rewrite(Def*, const Def*);
    undo_t analyze(Def*, const Def*);

    bool analyzed(const Def* def) {
        for (auto i = states_.rbegin(), e = states_.rend(); i != e; ++i) {
            if (i->analyzed.contains(def)) return true;
        }
        cur_state().analyzed.emplace(def);
        return false;
    }

    World& world_;
    std::vector<PassPtr> passes_;
    std::deque<State> states_;
    Nom2Nom reincanate_;

    template<class P> friend class Pass;
};

inline World& PassBase::world() { return man().world(); }

/// Inherit from this class using CRTP if you do need a Pass with a state.
template<class P>
class Pass : public PassBase {
public:
    Pass(PassMan& man, size_t index, const std::string& name)
        : PassBase(man, index, name)
    {}

    /// @name state-related getters
    //@{
    auto& states() { return man().states_; }
    auto& state(size_t i) { return *static_cast<typename P::State*>(states()[i].data[index()]); }
    auto& cur_state() { assert(!states().empty()); return *static_cast<typename P::State*>(states().back().data[index()]); }
    //@}
    /// @name search in the state stack
    //@{
    /// Searches states from back to top in the set @p S for @p key and puts it into @p S if not found.
    /// @return A triple: <tt> [iterator, undo, inserted] </tt>.
    template<class S>
    auto put(const typename S::key_type& key) {
        for (undo_t undo = states().size(); undo-- != 0;) {
            auto& set = std::get<S>(state(undo));
            if (auto i = set.find(key); i != set.end()) return std::tuple(i, undo, false);
        }

        auto [i, inserted] = std::get<S>(cur_state()).emplace(key);
        assert(inserted);
        return std::tuple(i, states().size()-1, true);
    }

    /// Searches states from back to top in the map @p M for @p key and inserts @p init if nothing is found.
    /// @return A triple: <tt> [iterator, undo, inserted] </tt>.
    template<class M>
    auto insert(const typename M::key_type& key, typename M::mapped_type&& init = {}) {
        for (undo_t undo = states().size(); undo-- != 0;) {
            auto& map = std::get<M>(state(undo));
            if (auto i = map.find(key); i != map.end()) return std::tuple(i, undo, false);
        }

        auto [i, inserted] = std::get<M>(cur_state()).emplace(key, std::move(init));
        assert(inserted);
        return std::tuple(i, states().size()-1, true);
    }
    //@}
    /// @name alloc/dealloc state
    //@{
    void* alloc() override { return new typename P::State(); }
    void dealloc(void* state) override { delete static_cast<typename P::State*>(state); }
    //@}
};

}
#endif
