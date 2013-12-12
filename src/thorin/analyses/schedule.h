#ifndef THORIN_ANALYSES_SCHEDULE_H
#define THORIN_ANALYSES_SCHEDULE_H

#include <vector>

#include "thorin/util/array.h"

namespace thorin {

class Scope;
class PrimOp;

typedef LambdaMap<std::vector<const PrimOp*>> Schedule;

Schedule schedule_early(const Scope&);
Schedule schedule_late(const Scope& scope);
Schedule schedule_smart(const Scope&);

} // namespace thorin

#endif
