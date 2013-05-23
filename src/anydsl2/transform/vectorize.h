#ifndef ANYDSL2_TRANSFORM_VECTORIZE_H
#define ANYDSL2_TRANSFORM_VECTORIZE_H

#include <cstdlib>

namespace anydsl2 {

class Def;
class Type;

const Type* vectorize(const Type* type, size_t length);
const Def* vectorize(const Def* cond, const Def* def);

}

#endif
