#ifndef THORIN_PASS_PASS_H
#define THORIN_PASS_PASS_H

#include "thorin/world.h"
#include <deque>

namespace thorin {

class PassMgr;

/// All Pass%es that want to be registered in the super @p PassMgr must implement this interface.
class Pass {
public:
    Pass(PassMgr& mgr)
        : mgr_(mgr)
    {}
    virtual ~Pass() {}

    PassMgr& mgr() { return mgr_; }
    virtual Def* rewrite(Def* nominal) { return nominal; }
    virtual const Def* rewrite(const Def*) = 0;
    virtual void analyze(const Def*) = 0;

private:
    PassMgr& mgr_;
};

/**
 * A super optimizer.
 * See "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
 */
class PassMgr {
public:
    PassMgr(World& world)
        : world_(world)
    {
        states_.emplace_back();
    }

    World& world() { return world_; }
    template<typename T, typename... Args>
    void create(Args&&... args) { passes_.emplace_back(std::make_unique<T>(*this, std::forward(args)...)); }
    void run();
    Def* rewrite(Def*);             ///< rewrites @em nominal @p Def%s
    const Def* rewrite(const Def*); ///< rewrites @em structural @p Def%s
    void analyze(const Def*);
    void undo(size_t u) { undo_ = std::min(undo_, u); }
    size_t num_states() const { return states_.size(); }

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

    template<class D> // D may be "Def" or "const Def"
    D* map(const Def* old_def, D* new_def) { cur_state().old2new.emplace(old_def, new_def); return new_def; }

    struct State {
        struct OrderLt { // visit basic blocks first
            bool operator()(Def* a, Def* b) { return a->type()->order() < b->type()->order(); }
        };

        State() = default;
        State(const State& other)
            : nominals(other.nominals)
            , old2new(other.old2new)
            , analyzed(other.analyzed)
        {}

        std::priority_queue<Def*, std::deque<Def*>, OrderLt> nominals;
        Def2Def old2new;
        DefSet analyzed;
        DefMap<Array<const Def*>> old_ops;
    };

    void enqueue(Def* nominal) { cur_state().nominals.push(nominal); }
    State& cur_state() { assert(!states_.empty()); return states_.back(); }
    void new_state() { states_.emplace_back(cur_state()); }

    static constexpr size_t No_Undo = std::numeric_limits<size_t>::max();

    World& world_;
    std::deque<std::unique_ptr<Pass>> passes_;
    std::deque<State> states_;
    size_t undo_ = No_Undo;
};

}

#endif
