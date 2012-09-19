#ifndef ANYDSL_ANALYSES_ORDER_H
#define ANYDSL_ANALYSES_ORDER_H

#include <boost/unordered_set.hpp>

#include "anydsl/util/array.h"

namespace anydsl {

class Lambda;

typedef boost::unordered_set<const Lambda*> LambdaSet;

void postorder(const LambdaSet& scope, const Lambda* entry);

} // namespace anydsl

#endif
