#include "anydsl/analyses/order.h"

#include "anydsl/lambda.h"

namespace anydsl {

static size_t number(const LambdaSet& scope, const Lambda* cur, size_t i) {
    // for each successor in scope
    for_all (succ, cur->succs()) {
        if (scope.find(succ) != scope.end() && succ->lid_invalid())
            i = number(scope, succ, i);
    }

    cur->lid = i;

    return i + 1;
}

void postorder(const LambdaSet& scope, const Lambda* entry) {
    for_all (lambda, scope)
        lambda->invalidate_lid();

    number(scope, entry, 0);
}

} // namespace anydsl
