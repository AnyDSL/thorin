#ifndef ANYDSL_ANALYSES_ROOTLAMBDAS_H
#define ANYDSL_ANALYSES_ROOTLAMBDAS_H

#include <boost/unordered_set.hpp>

namespace anydsl {

class Lambda;
class World;

typedef boost::unordered_set<const Lambda*> LambdaSet;

LambdaSet find_root_lambdas(const World& world);
LambdaSet find_root_lambdas(const LambdaSet& lambdas);

} // namespace anydsl

#endif
