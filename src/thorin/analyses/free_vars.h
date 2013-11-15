#ifndef ANYDSL2_ANALYSES_FREE_VARS_H
#define ANYDSL2_ANALYSES_FREE_VARS_H

#include <vector>

namespace anydsl2 {

class Param;
class Scope;

std::vector<Def> free_vars(const Scope&);

}

#endif
