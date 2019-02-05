#ifndef THORIN_OPT_OPTIMIZER_H
#define THORIN_OPT_OPTIMIZER_H

#include <list>
#include <memory>

namespace thorin {

class Def;
class Lam;
class World;

/// All Optimization%s that want to be registered in the super @p Optimizer must implement this interface.
class Optimization {
public:
    Optimization(const char* name)
        : name_(name)
    {}
    virtual ~Optimization() {}

    const char* name() const { return name_;}
    virtual void visit(Lam*) = 0;
    virtual void visit(const Def*) = 0;

private:
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

    World& world() const { return world_; }
    void add(std::unique_ptr<Optimization> opt) { optimizations_.emplace_back(std::move(opt)); }
    void run();

private:
    World& world_;
    std::list<std::unique_ptr<Optimization>> optimizations_;
};

}

#endif
