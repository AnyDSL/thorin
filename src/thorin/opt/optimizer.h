#ifndef THORIN_OPT_OPTIMIZER_H
#define THORIN_OPT_OPTIMIZER_H

#include "thorin/world.h"
#include "thorin/util/iterator.h"
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

class Context {
public:
    Def2Def old2new;
    DefSet analyzed; // TODO: merge with map above
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
    std::optional<const Def*> lookup(const Def* old_def) {
        for (auto&& ctxt : reverse_range(ctxts_)) {
            auto i = ctxt.old2new.find(old_def);
            auto e = ctxt.old2new.end();
            if (i == e) continue;

            // TODO path compression
            for (auto j = ctxt.old2new.find(i->second); j != e;) {
                auto tmp = j;
                j = ctxt.old2new.find(i->second);
                i = tmp;
            }

            return i->second;
        }
        return {};
    }

private:
    World& world_;
    std::deque<std::unique_ptr<Optimization>> opts_;
    std::queue<Def*> nominals_;
    std::deque<Context> ctxts_;
};

Optimizer std_optimizer(World&);

}

#endif
