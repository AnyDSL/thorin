#ifndef ANYDSL_ANALYSES_PLACEMENT_H
#define ANYDSL_ANALYSES_PLACEMENT_H

#include "anydsl/util/array.h"

namespace anydsl {

class Scope;
class PrimOp;

typedef Array< std::vector<const PrimOp*> > Places;
Places place(const Scope& sceop);

} // namespace anydsl

#endif
