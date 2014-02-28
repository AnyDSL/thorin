#ifndef THORIN_TRANSFORM_VECTORIZE_H
#define THORIN_TRANSFORM_VECTORIZE_H

#include <cstdlib>

namespace thorin {

class Lambda;

Lambda* vectorize(Scope&, size_t length);

}

#endif
