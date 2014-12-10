#ifndef THORIN_ANALYSES_SCHEDULE_H
#define THORIN_ANALYSES_SCHEDULE_H

#include <vector>

#include "thorin/analyses/scope.h"

namespace thorin {

class PrimOp;

class Schedule {
public:
    Schedule(const Scope& scope)
        : scope_(scope)
        , blocks_(scope.size())
    {}

    const Scope& scope() const { return scope_; }
    const std::vector<const PrimOp*>& operator [] (Lambda* lambda) const { return blocks_[scope().index(lambda)]; }

    typedef std::vector<std::vector<const PrimOp*>>::const_iterator const_iterator;
    const_iterator begin() const { return blocks_.begin(); }
    const_iterator end() const { return blocks_.end(); }

private:
    std::vector<const PrimOp*>& lookup(Lambda* lambda) { return blocks_[scope().index(lambda)]; }

    const Scope& scope_;
    std::vector<std::vector<const PrimOp*>> blocks_;

    friend const Schedule schedule_late(const Scope&);
    friend const Schedule schedule_smart(const Scope&);
};

const Schedule schedule_late(const Scope&);
const Schedule schedule_smart(const Scope&);

}

#endif
