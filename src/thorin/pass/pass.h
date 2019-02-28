#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include <deque>

#include "thorin/world.h"
#include "thorin/util/iterator.h"

namespace thorin {

class PassMgr;

/**
 * All Pass%es that want to be registered in the @p PassMgr must implement this interface.
 * However, inherit from @p Pass using CRTP to inherit some boilerplate regarding states for Pass%es.
 */
class PassBase {
public:
    PassBase(PassMgr& mgr, size_t id)
        : mgr_(mgr)
        , id_(id)
    {}
    virtual ~PassBase() {}

    /// @name getters
    //@{
    PassMgr& mgr() { return mgr_; }
    size_t id() const { return id_; }
    World& world();
    ///@}
    /// @name hooks for the PassMgr
    //@{
    virtual Def* rewrite(Def* nominal) { return nominal; }  ///< Rewrites @em nominal @p Def%s.
    virtual const Def* rewrite(const Def*) = 0;             ///< Rewrites @em structural @p Def%s.
    virtual void analyze(const Def*) {}                     ///< Invoked after the @p PassMgr has finisched @p rewrite%ing a nominal.
    ///@}
    /// @name alloc/dealloc state
    //@{
    virtual void* alloc() const { return nullptr; }
    virtual void dealloc(void*) const {}
    //@}

private:
    PassMgr& mgr_;
    size_t id_;
};

template<class P>
class Pass : public PassBase {
public:
    Pass(PassMgr& mgr, size_t pass_index)
        : PassBase(mgr, pass_index)
    {}

    /// Searches states from back to front in the map @p M for @p key using @p init if nothing is found.
    template<class M> auto& get(const typename M::key_type& key, typename M::mapped_type&& init);
    template<class M> auto& get(const typename M::key_type& key) { return get<M>(key, typename M::mapped_type()); }
    auto& states();     ///< Returns PassMgr::states_.
    auto& cur_state();  ///< Return PassMgr::states_.back().
    void* alloc() const override { return  new typename P::State(); }
    void dealloc(void* state) const override { delete (typename P::State*)state; }
};

/**
 * An optimizer that combines several optimizations in an optimal way.
 * See "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
 */
class PassMgr {
public:
    static constexpr size_t No_Undo = std::numeric_limits<size_t>::max();
    typedef std::unique_ptr<PassBase> PassPtr;

    PassMgr(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    template<typename P>
    PassMgr& create() { passes_.emplace_back(std::make_unique<P>(*this, passes_.size())); return *this; }
    void run();
    const Def* rebuild(const Def*); ///< Just performs the rebuild of a @em structural @p Def.
    void undo(size_t u) { undo_ = std::min(undo_, u); }
    size_t state_id() const { return states_.size(); }
    Def* cur_nominal() const { return cur_nominal_; }
    Lam* cur_lam() const { return cur_nominal_->as<Lam>(); }

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

    Def* rewrite(Def*);             ///< Rewrites @em nominal @p Def%s.
    const Def* rewrite(const Def*); ///< Rewrites @em structural @p Def%s.
    void analyze(const Def*);
    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { cur_state().old2new.emplace(old_def, new_def); return new_def; }

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

        struct OrderLt { // visit basic blocks first
            bool operator()(Def* a, Def* b) { return a->type()->order() < b->type()->order(); }
        };
        std::priority_queue<Def*, std::deque<Def*>, OrderLt> queue;
        Def2Def old2new;
        DefSet analyzed;
        Def* nominal;
        Array<const Def*> old_ops;
        const PassPtr* passes;
        Array<void*> data;
    };

    void new_state(Def* nominal, Defs old_ops) { states_.emplace_back(cur_state(), nominal, old_ops, passes_); }
    State& cur_state() { assert(!states_.empty()); return states_.back(); }

    World& world_;
    std::vector<PassPtr> passes_;
    std::deque<State> states_;
    Def* cur_nominal_;
    size_t undo_ = No_Undo;

    template<class P> friend auto& Pass<P>::states();
};

template<class P>
auto& Pass<P>::states() { return mgr().states_; }

template<class P>
auto& Pass<P>::cur_state() {
    assert(!states().empty());
    return *static_cast<typename P::State*>(states().back().data[id()]);
}

template<class P> template<class M>
auto& Pass<P>::get(const typename M::key_type& k, typename M::mapped_type&& init) {
    for (auto& state : reverse_range(states())) {
        auto& map = std::get<M>(*static_cast<typename P::State*>(state.data[id()]));
        if (auto i = map.find(k); i != map.end())
            return i->second;
    }

    return std::get<M>(cur_state()).emplace(k, std::move(init)).first->second;
}

}

#endif
