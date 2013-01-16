#ifndef ANYDSL2_ANALYSES_ROOTLAMBDAS_H
#define ANYDSL2_ANALYSES_ROOTLAMBDAS_H

#include "anydsl2/world.h"

namespace anydsl2 {

LambdaSet find_root_lambdas(const World& world);
LambdaSet find_root_lambdas(const LambdaSet& lambdas);

} // namespace anydsl2

#endif
