#ifndef THORIN_ANALYSES_SCHEDULE_H
#define THORIN_ANALYSES_SCHEDULE_H

#include "thorin/analyses/cfg.h"

namespace thorin {

template<bool> class DomTreeBase;
using DomTree = DomTreeBase<true>;

class Scheduler {
public:
    Scheduler() = default;
    explicit Scheduler(const Scope&);

    /// @name getters
    //@{
    const Scope& scope() const { return *scope_; }
    const F_CFG& cfg() const { return *cfg_; }
    const CFNode* cfg(Continuation* cont) const { return cfg()[cont]; }
    const DomTree& domtree() const { return *domtree_; }
    const Uses& uses(const Def* def) const { assert(def2uses_.contains(def)); return def2uses_.find(def)->second; }
    //@}

    /// @name compute schedules
    //@{
    Continuation* early(const Def*, DefSet* seen = nullptr);
    Continuation* late (const Def*);
    Continuation* smart(const Def*);
    //@}

    friend void swap(Scheduler& s1, Scheduler& s2) {
        using std::swap;
        swap(s1.scope_,    s2.scope_);
        swap(s1.cfg_,      s2.cfg_);
        swap(s1.domtree_,  s2.domtree_);
        swap(s1.early_,    s2.early_);
        swap(s1.late_,     s2.late_);
        swap(s1.smart_,    s2.smart_);
        swap(s1.def2uses_, s2.def2uses_);
    }

private:
    const Scope* scope_     = nullptr;
    const F_CFG* cfg_       = nullptr;
    const DomTree* domtree_ = nullptr;
    DefMap<Continuation*> early_;
    DefMap<Continuation*> late_;
    DefMap<Continuation*> smart_;
    DefMap<Uses> def2uses_;
};

using Schedule = std::vector<Continuation*>;
Schedule schedule(const Scope&);

}

#endif
