#ifndef THORIN_BE_AIR_H
#define THORIN_BE_AIR_H

#include "thorin/def.h"
#include "thorin/type.h"

namespace thorin {

void emit_thorin(const Scope&, bool fancy = true, bool colored = false);
void emit_thorin(World&, bool fancy = true, bool colored = false);
void emit_type(Type);
void emit_def(Def);
void emit_assignment(const PrimOp*);
void emit_head(const Lambda*);
void emit_jump(const Lambda*);

}

#endif
