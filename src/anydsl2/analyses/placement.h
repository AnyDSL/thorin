#ifndef ANYDSL2_ANALYSES_PLACEMENT_H
#define ANYDSL2_ANALYSES_PLACEMENT_H

#include <vector>

#include "anydsl2/util/array.h"

namespace anydsl2 {

class Def;
class Scope;
class PrimOp;

typedef std::vector<const PrimOp*> Schedule;
typedef Array<Schedule> Places;
Places place(const Scope& scope);
Places visit_early(const Scope&);
Places visit_late(const Scope&);

} // namespace anydsl2

#endif
