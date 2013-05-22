#ifndef ANYDSL2_TRANSFORM_VECTORIZE_H
#define ANYDSL2_TRANSFORM_VECTORIZE_H

#include <cstdlib>

namespace anydsl2 {

class Type;

const Type* vectorize(const Type* type, size_t length);

}

#endif
