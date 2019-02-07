#ifndef THORIN_OPT_OPTIMIZER_H
#define THORIN_OPT_OPTIMIZER_H

#include "thorin/world.h"
#include <deque>

namespace thorin {

class Optimizer;

/// All Optimization%s that want to be registered in the super @p Optimizer must implement this interface.
class Optimization {
public:
    Optimization(Optimizer& optimizer)
        : optimizer_(optimizer)
    {}
    virtual ~Optimization() {}

    Optimizer& optimizer() { return optimizer_; }
    virtual const Def* rewrite(const Def*) = 0;
    virtual void analyze(const Def*) = 0;

private:
    Optimizer& optimizer_;
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
    void enqueue(Def*);
    const Def* rewrite(const Def*);
    void analyze(const Def*);

private:
    World& world_;
    std::deque<std::unique_ptr<Optimization>> opts_;
    std::queue<Def*> nominals_;
    Def2Def old2new_;
    DefSet analyzed_; // TODO: merge with map above
};

Optimizer std_optimizer(World&);

}

#endif
