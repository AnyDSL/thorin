#ifndef THORIN_ANALYSES_SCHEDULE_H
#define THORIN_ANALYSES_SCHEDULE_H

#include "thorin/analyses/cfg.h"

namespace thorin {

template<bool> class DomTreeBase;
using DomTree = DomTreeBase<true>;

class Schedule {
public:
    Schedule(const Scope&);

    /// @name getters
    //@{
    const Scope& scope() const { return scope_; }
    const F_CFG& cfg() const { return cfg_; }
    const CFNode* cfg(Continuation* cont) const { return cfg()[cont]; }
    const DomTree& domtree() const { return domtree_; }
    const Uses& uses(const Def* def) const { return def2uses_.find(def)->second; }
    //@}

    /// @name compute schedules
    //@{
    Continuation* schedule_early(const Def*);
    Continuation* schedule_late (const Def*);
    Continuation* schedule_smart(const Def*);
    //@}

private:
    void compute_def2uses();

    const Scope& scope_;
    const F_CFG& cfg_;
    const DomTree& domtree_;
    DefMap<Continuation*> early_;
    DefMap<Continuation*> late_;
    DefMap<Continuation*> smart_;
    DefMap<Uses> def2uses_;
};

using BlockSchedule = std::vector<Continuation*>;
BlockSchedule block_schedule(const Scope&);

}

#endif
