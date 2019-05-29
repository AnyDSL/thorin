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
    static constexpr size_t No_Undo = std::numeric_limits<size_t>::max();

    PassBase(PassMan& man, size_t index)
        : man_(man)
        , index_(index)
    {}
    virtual ~PassBase() {}

    /// @name getters
    //@{
    PassMan& man() { return man_; }
    size_t index() const { return index_; }
    World& world();
    ///@}
    /// @name hooks for the PassMan
    //@{
    virtual const Def* rewrite(const Def*) = 0; ///< Rewrites @em structural @p Def%s.
    virtual void inspect(Def*) {}               ///< Inspects a @em nominal @p Def when first encountering it.
    virtual void enter(Def*) {}                 ///< Invoked when a @em nominal is first time the top of the PassMan::queue().
    /**
     * Invoked after the @p PassMan has finished @p rewrite%ing a nominal.
     * Return the state id to rollback to or @p No_Undo if no undo is required.
     */
    virtual size_t analyze(const Def*) { return No_Undo; }
    ///@}
    /// @name mangage state - dummy implementations here
    //@{
    virtual void* alloc() { return nullptr; }
    virtual void dealloc(void*) {}
    //@}

private:
    PassMan& man_;
    size_t index_;
};

/**
 * An optimizer that combines several optimizations in an optimal way.
 * This is loosely based upon:
 * "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
 */
class PassMan {
public:
    typedef std::unique_ptr<PassBase> PassPtr;

    PassMan(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    template<class P, class... Args>
    PassMan& create(Args... args) { passes_.emplace_back(std::make_unique<P>(*this, passes_.size()), std::forward<Args>(args)...); return *this; }
    void run();
    size_t cur_state_id() const { return states_.size(); }
    Def* cur_nominal() const { return cur_nominal_; }
    Lam* cur_lam() const { return cur_nominal()->as<Lam>(); }
    void new_state() { states_.emplace_back(cur_state(), passes_); }
    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { cur_state().old2new[old_def] = new_def; return new_def; }
    bool push(const Def*); ///< Pushes to @p rewrite stack.
    void rewrite();

    std::optional<const Def*> lookup(const Def* old_def) {
        auto& old2new = cur_state().old2new;
        if (auto i = old2new.find(old_def); i != old2new.end()) return lookup(old2new, i);
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
        State(const State& prev, const std::vector<PassPtr>& passes)
            : queue(prev.queue)
            , stack(prev.stack)
            , old2new(prev.old2new)
            , analyzed(prev.analyzed)
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

        std::deque<std::tuple<Def*, Array<const Def*>>> nominals;
        Queue queue;
        std::stack<const Def*> stack;
        Def2Def old2new;
        DefSet analyzed;
        const PassPtr* passes;
        Array<void*> data;
    };

    size_t analyze(const Def*, size_t);
    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    const State& cur_state() const { assert(!states_.empty()); return states_.back(); }
    State::Queue& queue() { return cur_state().queue; }
    std::stack<const Def*>& stack() { return cur_state().stack; }

    World& world_;
    std::vector<PassPtr> passes_;
    size_t time_ = 0;
    std::deque<State> states_;
    Def* cur_nominal_ = nullptr;

    template<class P> friend class Pass;
};

inline World& PassBase::world() { return man().world(); }

/// Inherit from this class using CRTP if you do need a Pass with a state.
template<class P>
class Pass : public PassBase {
public:
    Pass(PassMan& man, size_t index)
        : PassBase(man, index)
    {}

    /// @name state-related getters
    //@{
    auto& states() { return man().states_; }
    auto& state(size_t i) { return *static_cast<typename P::State*>(states()[i].data[index()]); }
    auto& cur_state() { assert(!states().empty()); return *static_cast<typename P::State*>(states().back().data[index()]); }
    //@}
    /// @name recursive search in the state stack
    //@{
    /// Searches states from back to front in the map @p M for @p key using @p init if nothing is found.
    template<class M>
    auto& get(const typename M::key_type& key, typename M::mapped_type&& init) {
        auto& map = std::get<M>(cur_state());
        if (auto i = map.find(key); i != map.end())
            return i->second;

        return std::get<M>(cur_state()).emplace(key, std::move(init)).first->second;
    }
    /// Same as above but uses the default constructor as init.
    template<class M>
    auto& get(const typename M::key_type& key) { return get<M>(key, typename M::mapped_type()); }
    //@}
    /// @name alloc/dealloc state
    //@{
    void* alloc() override { return states().empty() ? new typename P::State() : new typename P::State(cur_state()); }
    void dealloc(void* state) override { delete static_cast<typename P::State*>(state); }
    //@}
};

}

#endif
