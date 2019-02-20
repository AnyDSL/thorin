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
    {}

    World& world() { return world_; }
    template<typename T, typename... Args>
    void create(Args&&... args) { passes_.emplace_back(std::make_unique<T>(*this, std::forward(args)...)); }
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

private:
    const Def* lookup(Def2Def::iterator i) {
        if (auto j = old2new_.find(i->second); j != old2new_.end())
            i->second = lookup(j); // path compression + transitive replacements
        return i->second;
    }

    // visit basic blocks first
    struct OrderLt {
        bool operator()(Def* a, Def* b) { return a->type()->order() < b->type()->order(); }
    };

    World& world_;
    std::deque<std::unique_ptr<Pass>> passes_;
    std::priority_queue<Def*, std::deque<Def*>, OrderLt> nominals_;
    Def2Def old2new_;
    DefSet analyzed_;
};

}

#endif
