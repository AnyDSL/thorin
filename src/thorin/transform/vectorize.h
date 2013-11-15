#ifndef ANYDSL2_TRANSFORM_VECTORIZE_H
#define ANYDSL2_TRANSFORM_VECTORIZE_H

#include <cstdlib>

namespace anydsl2 {

class Scope;
class Lambda;

Lambda* vectorize(Scope& scope, size_t length);

}

#endif
