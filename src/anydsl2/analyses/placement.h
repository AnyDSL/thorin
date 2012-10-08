#ifndef ANYDSL_ANALYSES_PLACEMENT_H
#define ANYDSL_ANALYSES_PLACEMENT_H

#include "anydsl2/util/array.h"

namespace anydsl2 {

class Scope;
class PrimOp;

typedef Array< std::vector<const PrimOp*> > Places;
Places place(const Scope& sceop);

} // namespace anydsl2

#endif
