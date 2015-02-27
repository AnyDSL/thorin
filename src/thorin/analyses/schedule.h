#ifndef THORIN_ANALYSES_SCHEDULE_H
#define THORIN_ANALYSES_SCHEDULE_H

#include <vector>

#include "thorin/analyses/cfg.h"

namespace thorin {

class PrimOp;

class Schedule {
public:
    typedef F_CFG::Map<std::vector<const PrimOp*>> Blocks;

    Schedule(const Scope& scope)
        : scope_(scope)
        , blocks_(*scope.cfa()->f_cfg())
    {}

    const Scope& scope() const { return scope_; }
    const std::vector<const PrimOp*>& operator [] (Lambda* lambda) const { return const_cast<Schedule*>(this)->lookup(lambda); }

    typedef Blocks::const_iterator const_iterator;
    const_iterator begin() const { return blocks_.begin(); }
    const_iterator end() const { return blocks_.end(); }

private:
    std::vector<const PrimOp*>& lookup(Lambda* lambda) { return blocks_[scope().cfa()->lookup(lambda)]; }

    const Scope& scope_;
    Blocks blocks_;

    friend const Schedule schedule_late(const Scope&);
    friend const Schedule schedule_smart(const Scope&);
};

const Schedule schedule_late(const Scope&);
const Schedule schedule_smart(const Scope&);

}

#endif
