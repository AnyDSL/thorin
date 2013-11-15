#ifndef THORIN_ANALYSES_PLACEMENT_H
#define THORIN_ANALYSES_PLACEMENT_H

#include <vector>

#include "thorin/util/array.h"

namespace thorin {

class Scope;
class PrimOp;

typedef Array<std::vector<const PrimOp*>> Schedule;

Schedule schedule_early(const Scope&);
Schedule schedule_late(const Scope&, size_t& pass);
inline Schedule schedule_late(const Scope& scope) { size_t pass; return schedule_late(scope, pass); }
Schedule schedule_smart(const Scope&);

} // namespace thorin

#endif
