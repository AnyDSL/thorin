#ifndef THORIN_ANALYSES_FREE_VARS_H
#define THORIN_ANALYSES_FREE_VARS_H

#include <vector>

namespace thorin {

class Param;
template<bool> class ScopeBase;

std::vector<Def> free_vars(const ScopeBase<true>&);

}

#endif
