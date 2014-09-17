#ifndef THORIN_ANALYSES_TOP_LEVEL_SCOPES_H
#define THORIN_ANALYSES_TOP_LEVEL_SCOPES_H

#include <functional>

#include "thorin/util/autoptr.h"
#include "thorin/analyses/scope.h"

namespace thorin {

AutoVector<Scope*> top_level_scopes(World&);
void top_level_scopes(World&, std::function<void(Scope&)>, bool elide_empty = true);

}

#endif
