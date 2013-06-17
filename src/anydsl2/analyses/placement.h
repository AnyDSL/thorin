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
std::vector<const Def*> visit_early(const Scope&);
std::vector<const Def*> visit_late(const Scope&);

} // namespace anydsl2

#endif
