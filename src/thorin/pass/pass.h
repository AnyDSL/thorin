#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include <deque>

#include "thorin/world.h"
#include "thorin/util/iterator.h"

namespace thorin {

class PassMan;

/**
 * All Pass%es that want to be registered in the @p PassMan must implement this interface.
 * * Inherit from this class if your pass doesn't need state.
 * * Inherit from PassBase using CRTP if you do need state.
 */
class PassBase {
public:
    PassBase(PassMan& man, size_t id)
        : man_(man)
        , id_(id)
    {}
    virtual ~PassBase() {}

    /// @name getters
    //@{
    PassMan& man() { return man_; }
    size_t id() const { return id_; }
    World& world();
    ///@}
    /// @name hooks for the PassMan
    //@{
    virtual Def* rewrite(Def* nominal) { return nominal; }  ///< Rewrites @em nominal @p Def%s.
    virtual const Def* rewrite(const Def*) = 0;             ///< Rewrites @em structural @p Def%s.
    virtual void analyze(const Def*) {}                     ///< Invoked after the @p PassMan has finisched @p rewrite%ing a nominal.
    ///@}
    /// @name alloc/dealloc state - dummy implementations here
    //@{
    virtual void* alloc() const { return nullptr; }
    virtual void dealloc(void*) const {}
    //@}

private:
    PassMan& man_;
    size_t id_;
};

/**
 * An optimizer that combines several optimizations in an optimal way.
 * See "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
 */
class PassMan {
public:
    static constexpr size_t No_Undo = std::numeric_limits<size_t>::max();
    typedef std::unique_ptr<PassBase> PassPtr;

    PassMan(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    template<typename P>
    PassMan& create() { passes_.emplace_back(std::make_unique<P>(*this, passes_.size())); return *this; }
    void run();
    const Def* rebuild(const Def*); ///< Just performs the rebuild of a @em structural @p Def.
    void undo(size_t u) { undo_ = std::min(undo_, u); }
    size_t state_id() const { return states_.size(); }
    Def* cur_nominal() const { return cur_nominal_; }
    Lam* cur_lam() const { return cur_nominal()->as<Lam>(); }
    void new_state() { states_.emplace_back(cur_state(), cur_nominal(), cur_nominal()->ops(), passes_); }

    std::optional<const Def*> lookup(const Def* old_def) {
        auto& old2new = cur_state().old2new;
        if (auto i = old2new.find(old_def); i != old2new.end())
            return lookup(old2new, i);
        return {};
    }

private:
    static const Def* lookup(Def2Def& old2new, Def2Def::iterator i) {
        if (auto j = old2new.find(i->second); j != old2new.end() && i != j)
            i->second = lookup(old2new, j); // path compression + transitive replacements
        return i->second;
    }

    struct State {
        State() = default;
        State(const State&) = delete;
        State(State&&) = delete;
        State& operator=(State) = delete;

        State(const std::vector<PassPtr>& passes)
            : passes(passes.data())
            , data(passes.size(), [&](auto i) { return passes[i]->alloc(); })
        {}
        State(const State& prev, Def* nominal, Defs old_ops, const std::vector<PassPtr>& passes)
            : queue(prev.queue)
            , old2new(prev.old2new)
            , analyzed(prev.analyzed)
            , nominal(nominal)
            , old_ops(old_ops)
            , passes(passes.data())
            , data(passes.size(), [&](auto i) { return passes[i]->alloc(); })
        {}
        ~State() {
            for (size_t i = 0, e = data.size(); i != e; ++i)
                passes[i]->dealloc(data[i]);
        }

        typedef std::tuple<Def*, size_t> Item;
        struct OrderLt {
            // visit basic blocks first, then sort by time stamp to make it stable
            bool operator()(Item a, Item b) {
                return std::get<0>(a)->type()->order() != std::get<0>(b)->type()->order()
                     ? std::get<0>(a)->type()->order() >  std::get<0>(b)->type()->order()
                     : std::get<1>(a)                  >  std::get<1>(b);
            }
        };
        typedef std::priority_queue<Item, std::deque<Item>, OrderLt> Queue;

        Queue queue;
        Def2Def old2new;
        DefSet analyzed;
        Def* nominal;
        Array<const Def*> old_ops;
        const PassPtr* passes;
        Array<void*> data;
    };

    Def* rewrite(Def*);             ///< Rewrites @em nominal @p Def%s.
    const Def* rewrite(const Def*); ///< Rewrites @em structural @p Def%s.
    void analyze(const Def*);
    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { cur_state().old2new.emplace(old_def, new_def); return new_def; }
    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    State::Queue& queue() { return cur_state().queue; }

    World& world_;
    std::vector<PassPtr> passes_;
    size_t undo_ = No_Undo;
    size_t time_ = 0;
    std::deque<State> states_;
    Def* cur_nominal_ = nullptr;

    template<class P> friend class Pass;
};

/// Inherit from this class using CRTP if you do need a Pass with a state.
template<class P>
class Pass : public PassBase {
public:
    Pass(PassMan& man, size_t id)
        : PassBase(man, id)
    {}

    /// @name getters
    //@{
    /// Returns PassMan::states_.
    auto& states() { return man().states_; }
    /// Return PassMan::states_.back().
    auto& cur_state() { assert(!states().empty()); return *static_cast<typename P::State*>(states().back().data[id()]); }
    //@}
    /// @name recursive search in the state stack
    //@{
    /// Searches states from back to front in the map @p M for @p key using @p init if nothing is found.
    template<class M>
    auto& get(const typename M::key_type& key, typename M::mapped_type&& init) {
        for (auto& state : reverse_range(states())) {
            auto& map = std::get<M>(*static_cast<typename P::State*>(state.data[id()]));
            if (auto i = map.find(key); i != map.end())
                return i->second;
        }

        return std::get<M>(cur_state()).emplace(key, std::move(init)).first->second;
    }
    /// Same as above but uses the default constructor as init.
    template<class M>
    auto& get(const typename M::key_type& key) { return get<M>(key, typename M::mapped_type()); }
    //@}
    /// @name alloc/dealloc state
    //@{
    void* alloc() const override { return new typename P::State(); }
    void dealloc(void* state) const override { delete static_cast<typename P::State*>(state); }
    //@}
};

}

#endif
