#ifndef THORIN_OPT_OPTIMIZER_H
#define THORIN_OPT_OPTIMIZER_H

#include "thorin/world.h"
#include <deque>

namespace thorin {

class Optimizer;

/// All Optimization%s that want to be registered in the super @p Optimizer must implement this interface.
class Optimization {
public:
    Optimization(Optimizer& optimizer, const char* name)
        : optimizer_(optimizer)
        , name_(name)
    {}
    virtual ~Optimization() {}

    const char* name() const { return name_;}
    Optimizer& optimizer() { return optimizer_; }
    virtual void enter(Lam*) = 0;
    virtual const Def* visit(const Def*) = 0;

private:
    Optimizer& optimizer_;
    const char* name_;
};

/**
 * A super optimizer.
 * See "Composing dataflow analyses and transformations" by Lerner, Grove, Chambers.
 */
class Optimizer {
public:
    Optimizer(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    template<typename T, typename... Args>
    void create(Args&&... args) { opts_.emplace_back(std::make_unique<T>(*this, std::forward(args)...)); }
    void run();

    const Def* lookup(const Def* old) {
        // TODO path compression
        while (auto def = def2def_.lookup(old)) {
            if (*def == old) break;
            old = *def;
        }
        return old;
    }

private:
    World& world_;
    std::deque<std::unique_ptr<Optimization>> opts_;
    unique_queue<LamSet> lams_;
    unique_stack<DefSet> defs_;
    Def2Def def2def_;

    friend void swap(Optimizer& a, Optimizer& b);
};

Optimizer std_optimizer(World&);

}

#endif
