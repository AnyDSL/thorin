#ifndef THORIN_TRANSFORM_VECTORIZE_H
#define THORIN_TRANSFORM_VECTORIZE_H

#include <cstdlib>

namespace thorin {

class Scope;
class Lambda;

Lambda* vectorize(Scope& scope, size_t length);

}

#endif
