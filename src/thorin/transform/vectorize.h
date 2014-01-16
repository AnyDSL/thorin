#ifndef THORIN_TRANSFORM_VECTORIZE_H
#define THORIN_TRANSFORM_VECTORIZE_H

#include <cstdlib>

namespace thorin {

template<bool> class ScopeBase;
class Lambda;

Lambda* vectorize(Lambda*, size_t length);

}

#endif
