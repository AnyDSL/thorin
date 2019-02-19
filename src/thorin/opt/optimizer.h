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
    virtual Def* rewrite(Def* nominal) { return nominal; }
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
    Def* rewrite(Def*);             ///< rewrites @em nominal @p Def%s
    const Def* rewrite(const Def*); ///< rewrites @em structural @p Def%s
    void analyze(const Def*);

    std::optional<const Def*> lookup(const Def* old_def) {
        if (auto i = old2new_.find(old_def); i != old2new_.end())
            return lookup(i);
        return {};
    }

    const Def* lookup(Def2Def::iterator i) {
        if (auto j = old2new_.find(i->second); j != old2new_.end())
            i->second = lookup(j); // path compression + transitive replacements
        return i->second;
    }

private:
    World& world_;
    std::deque<std::unique_ptr<Optimization>> opts_;
    std::queue<Def*> nominals_;
    Def2Def old2new_;
    DefSet analyzed_;
};

Optimizer std_optimizer(World&);

}

#endif
