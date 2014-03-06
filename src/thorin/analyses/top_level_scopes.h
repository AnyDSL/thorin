#ifndef THORIN_ANALYSES_TOP_LEVEL_H
#define THORIN_ANALYSES_TOP_LEVEL_H

#include "thorin/analyses/scope.h"
#include "thorin/util/autoptr.h"

namespace thorin {

AutoVector<Scope*> top_level_scopes(World&);

}

#endif
