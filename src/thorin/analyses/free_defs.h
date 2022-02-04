#ifndef THORIN_ANALYSES_FREE_DEFS_H
#define THORIN_ANALYSES_FREE_DEFS_H

#include "thorin/continuation.h"

namespace thorin {

class Scope;

DefSet free_defs(const Scope&, bool include_closures = true);
DefSet free_defs(Lam* entry);

}

#endif
