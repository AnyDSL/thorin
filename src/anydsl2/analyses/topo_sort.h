#ifndef ANYDSL2_ANALYSES_TOPO_SORT_H
#define ANYDSL2_ANALYSES_TOPO_SORT_H

#include <vector>

namespace anydsl2 {

class Def;
class Scope;

std::vector<const Def*> topo_sort(const Scope&);

}

#endif
