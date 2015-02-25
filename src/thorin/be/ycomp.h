#ifndef YCOMP_BE_H
#define YCOMP_BE_H

#include "thorin/def.h"
#include "thorin/type.h"

namespace thorin {

class Scope;

void emit_ycomp(const Scope&, int indent = 0, bool fancy = true, bool colored = false);
void emit_ycomp(const World&, bool fancy = true, bool colored = false);
void emit_ycomp_def(Def);

}

#endif
