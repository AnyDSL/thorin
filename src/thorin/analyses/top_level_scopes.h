#ifndef THORIN_ANALYSES_TOP_LEVEL_SCOPES_H
#define THORIN_ANALYSES_TOP_LEVEL_SCOPES_H

#include <functional>

#include "thorin/analyses/scope.h"

namespace thorin {

template<bool elide_empty = true>
void top_level_scopes(World&, std::function<void(const Scope&)>);

}

#endif
