#ifndef THORIN_ANALYSES_TOP_LEVEL_H
#define THORIN_ANALYSES_TOP_LEVEL_H

#include "thorin/lambda.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class Scope;

AutoVector<Scope*> top_level_scopes(World&);

}

#endif
