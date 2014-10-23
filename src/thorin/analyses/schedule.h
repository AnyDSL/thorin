#ifndef THORIN_ANALYSES_SCHEDULE_H
#define THORIN_ANALYSES_SCHEDULE_H

#include <vector>

#include "thorin/analyses/scope.h"

namespace thorin {

class PrimOp;

typedef LambdaMap<std::vector<const PrimOp*>> Schedule;

Schedule schedule_early(const Scope&);
Schedule schedule_late(const Scope&);
Schedule schedule_smart(const Scope&);

}

#endif
