#ifndef THORIN_ANALYSES_FREE_VARS_H
#define THORIN_ANALYSES_FREE_VARS_H

#include <vector>

#include "thorin/def.h"

namespace thorin {

class Scope;

/// @deprecated { use \p free_params instead }
std::vector<const Def*> free_vars(const Scope&);

}

#endif
