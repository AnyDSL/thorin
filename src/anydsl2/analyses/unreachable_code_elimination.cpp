#include "anydsl2/analyses/unreachable_code_elimination.h"

#include "anydsl2/lambda.h"

namespace anydsl2 {

static void uce_insert(LambdaSet& lambdas, Lambda* lambda) {
    assert(lambdas.find(lambda) != lambdas.end() && "not in set");

    if (lambda->is_marked()) return;
    lambda->mark();

    for_all (succ, lambda->succs())
        if (lambdas.find(succ) != lambdas.end())
            uce_insert(lambdas, succ);
}

void unreachable_code_elimination(LambdaSet& lambdas, ArrayRef<Lambda*> reachable) {
    for_all (lambda, lambdas) 
        lambda->unmark(); 

    for_all (lambda, reachable)
        uce_insert(lambdas, lambda);

    for (LambdaSet::iterator i = lambdas.begin(); i != lambdas.end();) {
        LambdaSet::iterator j = i++;
        Lambda* lambda = *j;

        if (!lambda->is_marked())
            lambdas.erase(j);
    }
}

void unreachable_code_elimination(LambdaSet& lambdas, Lambda* reachable) {
    Lambda* array[1] = { reachable };
    unreachable_code_elimination(lambdas, array);
}

} // namespace anydsl2
