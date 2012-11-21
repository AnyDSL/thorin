#ifndef ANYDSL2_ANALYSES_PLACEMENT_H
#define ANYDSL2_ANALYSES_PLACEMENT_H

#include "anydsl2/util/array.h"

namespace anydsl2 {

class Scope;
class PrimOp;

typedef std::vector<const PrimOp*> Schedule;
typedef Array<Schedule> Places;
Places place_early(const Scope& scope);
Places place_late(const Scope& scope);

} // namespace anydsl2

#endif
