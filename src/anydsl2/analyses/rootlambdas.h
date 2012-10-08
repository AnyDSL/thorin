#ifndef ANYDSL2_ANALYSES_ROOTLAMBDAS_H
#define ANYDSL2_ANALYSES_ROOTLAMBDAS_H

#include <boost/unordered_set.hpp>

namespace anydsl2 {

class Lambda;
class World;

typedef boost::unordered_set<Lambda*> LambdaSet;

LambdaSet find_root_lambdas(const World& world);
LambdaSet find_root_lambdas(const LambdaSet& lambdas);

} // namespace anydsl2

#endif
