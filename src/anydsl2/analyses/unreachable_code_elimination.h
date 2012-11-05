#ifndef ANYDSL2_UNREACHABLE_CODE_ELIMINATION_H
#define ANYDSL2_UNREACHABLE_CODE_ELIMINATION_H

#include "anydsl2/lambda.h"
#include "anydsl2/util/array.h"

namespace anydsl2 {

void unreachable_code_elimination(LambdaSet& lambdas, ArrayRef<Lambda*> reachable);
void unreachable_code_elimination(LambdaSet& lambdas, Lambda* reachable);

} // namespace anydsl2

#endif
