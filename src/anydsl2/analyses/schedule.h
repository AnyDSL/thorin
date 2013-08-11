#ifndef ANYDSL2_ANALYSES_PLACEMENT_H
#define ANYDSL2_ANALYSES_PLACEMENT_H

#include <vector>

#include "anydsl2/util/array.h"

namespace anydsl2 {

class Scope;
class PrimOp;

typedef Array<std::vector<const PrimOp*>> Schedule;

Schedule schedule_early(const Scope&);
Schedule schedule_late(const Scope&, size_t& pass);
inline Schedule schedule_late(const Scope& scope) { size_t pass; return schedule_late(scope, pass); }
Schedule schedule_smart(const Scope&);

} // namespace anydsl2

#endif
