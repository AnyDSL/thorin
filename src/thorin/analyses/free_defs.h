#ifndef THORIN_ANALYSES_FREE_DEFS_H
#define THORIN_ANALYSES_FREE_DEFS_H

#include "thorin/continuation.h"

namespace thorin {

class Scope;

DefSet spillable_free_defs(const Scope&);

DefSet spillable_free_defs(ScopesForest&, Continuation*);

}

#endif
