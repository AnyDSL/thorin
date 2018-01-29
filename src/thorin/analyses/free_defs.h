#ifndef THORIN_ANALYSES_FREE_DEFS_H
#define THORIN_ANALYSES_FREE_DEFS_H

#include "thorin/continuation.h"

namespace thorin {

class Scope;

DefSet free_defs(const Scope&, bool include_closures = true);
DefSet free_defs(Continuation* entry);


/// Returns @c true if @p entry has @p free_defs that are @em not @p Continuation%s.
bool has_free_vars(Continuation* entry);

}

#endif
