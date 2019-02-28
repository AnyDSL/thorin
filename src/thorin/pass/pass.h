#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include <deque>

#include "thorin/world.h"
#include "thorin/util/iterator.h"

namespace thorin {

class PassMgr;

/**
 * All Pass%es that want to be registered in the @p PassMgr must implement this interface.
 * However, inherit from @p Pass using CRTP to inherit some boilerplate
 */
class PassBase {
public:
    PassBase(PassMgr& mgr, size_t id)
        : mgr_(mgr)
        , id_(id)
    {}
    virtual ~PassBase() {}

    PassMgr& mgr() { return mgr_; }
    size_t id() const { return id_; }
    World& world();
    virtual Def* rewrite(Def* nominal) { return nominal; }  ///< Rewrites @em nominal @p Def%s.
    virtual const Def* rewrite(const Def*) = 0;             ///< Rewrites @em structural @p Def%s.
    virtual void analyze(const Def*) {}                     ///< Invoked after the @p PassMgr has finisched @p rewrite%ing a nominal.

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

    template<class M> auto& get(const typename M::key_type&, typename M::mapped_type&&);
    static void* creator() { return std::is_empty<typename P::State>::value ? nullptr : new typename P::State(); }
    static void deleter(void* state) { if (!std::is_empty<typename P::State>::value) delete (typename P::State*)state; }
};

/**
 * An optimizer that combines several optimizations in an optimal way.
 * See "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
 */
class PassMgr {
public:
    static constexpr size_t No_Undo = std::numeric_limits<size_t>::max();
    using Creator = void*(*)();
    using Deleter = void(*)(void*);
    using PassData = std::tuple<std::unique_ptr<PassBase>, Creator, Deleter>;

    PassMgr(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    template<typename P>
    PassMgr& create() { passes_.emplace_back(std::tuple(std::make_unique<P>(*this, passes_.size()), P::creator, P::deleter)); return *this; }
    void run();
    const Def* rebuild(const Def*); ///< just performs the rebuild of a @em struct @p Def
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

    Def* rewrite(Def*);             ///< rewrites @em nominal @p Def%s
    const Def* rewrite(const Def*); ///< rewrites @em structural @p Def%s
    void analyze(const Def*);
    void enqueue(Def* nominal) { cur_state().queue.push(nominal); }
    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { cur_state().old2new.emplace(old_def, new_def); return new_def; }

    struct State {
        struct OrderLt { // visit basic blocks first
            bool operator()(Def* a, Def* b) { return a->type()->order() < b->type()->order(); }
        };

        State() = default;
        State(const State&) = delete;
        State(State&&) = delete;
        State& operator=(State) = delete;

        State(const std::vector<PassData>& pass_data)
            : pass_data(pass_data.data())
            , pass_states(pass_data.size(), [&](auto i) { return std::get<Creator>(pass_data[i])(); })
        {}
        State(const State& prev, Def* nominal, Defs old_ops, const std::vector<PassData>& pass_data)
            : queue(prev.queue)
            , old2new(prev.old2new)
            , analyzed(prev.analyzed)
            , nominal(nominal)
            , old_ops(old_ops)
            , pass_data(pass_data.data())
            , pass_states(pass_data.size(), [&](auto i) { return std::get<Creator>(pass_data[i])(); })
        {}
        ~State() {
            for (size_t i = 0, e = pass_states.size(); i != e; ++i)
                std::get<Deleter>(pass_data[i])(pass_states[i]);
        }

        std::priority_queue<Def*, std::deque<Def*>, OrderLt> queue;
        Def2Def old2new;
        DefSet analyzed;
        Def* nominal;
        Array<const Def*> old_ops;
        const PassData* pass_data;
        Array<void*> pass_states;
    };

    void new_state(Def* nominal, Defs old_ops) { states_.emplace_back(cur_state(), nominal, old_ops, passes_); }
    State& cur_state() { assert(!states_.empty()); return states_.back(); }

    World& world_;
    std::vector<PassData> passes_;
    std::deque<State> states_;
    Def* cur_nominal_;
    size_t undo_ = No_Undo;

    template<class P> template<class M> friend auto& Pass<P>::get(const typename M::key_type&, typename M::mapped_type&&);
};

template<class P> template<class M>
auto& Pass<P>::get(const typename M::key_type& k, typename M::mapped_type&& m) {
    for (auto& state : reverse_range(mgr().states_)) {
        auto& map = std::get<M>(*static_cast<typename P::State*>(state.pass_states[id()]));
        if (auto i = map.find(k); i != map.end())
            return i->second;
    }

    assert(!mgr().states_.empty());
    auto& map = std::get<M>(*static_cast<typename P::State*>(mgr().states_.back().pass_states[id()]));
    return map.emplace(k, std::move(m)).first->second;
}

}

#endif
