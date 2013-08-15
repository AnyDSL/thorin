#ifndef ANYDSL2_BE_AIR_H
#define ANYDSL2_BE_AIR_H

#include "anydsl2/anydsl_fwd.h"

namespace anydsl2 {

void emit_air(World&, bool fancy = false);
void emit_type(const Type*);
void emit_def(const Def*);
void emit_assignment(const PrimOp*);
void emit_head(const Lambda*);
void emit_jump(const Lambda*);

}

#endif
