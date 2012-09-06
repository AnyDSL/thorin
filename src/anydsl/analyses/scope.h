#ifndef ANYDSL_ANALYSES_SCOPE_H
#define ANYDSL_ANALYSES_SCOPE_H

#include <boost/unordered_set.hpp>

namespace anydsl {

class Lambda;

typedef boost::unordered_set<const Lambda*> LambdaSet;

LambdaSet find_scope(const Lambda* labda);

} // namespace anydsl

#endif

