#ifndef ANYDSL2_ANALYSES_ROOTLAMBDAS_H
#define ANYDSL2_ANALYSES_ROOTLAMBDAS_H

#include <vector>

namespace anydsl2 {

class Lambda;
class World;

std::vector<Lambda*> find_root_lambdas(World& world);

} // namespace anydsl2

#endif
