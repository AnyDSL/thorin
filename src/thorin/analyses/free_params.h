#ifndef THORIN_ANALYSES_FREE_PARAMS_H
#define THORIN_ANALYSES_FREE_PARAMS_H

#include "thorin/continuation.h"

namespace thorin {

class Scope;

DefSet free_params(const Scope&);

}

#endif
