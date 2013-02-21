#include "anydsl2/transform/merge_lambdas.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/domtree.h"

namespace anydsl2 {

void merge_lambdas(World& world) {
    std::vector<Lambda*> todo;
    for_all (top, find_root_lambdas(world)) {
        Scope scope(top);
        for (size_t i = scope.size(); i-- != 0;) {
            Lambda* lambda = scope.rpo(i);
            if (lambda->num_uses() == 1 && !lambda->attr().is_extern()) {
                Use use = *lambda->uses().begin();
                if (use.index() == 0) {
                    Lambda* ulambda = use.def()->as_lambda();
                    if (scope.contains(ulambda) && scope.domtree().dominates(ulambda, lambda))
                        todo.push_back(lambda);
                }
            }
        }
    }

    for_all (lambda, todo) {
        Scope scope(lambda);
        Lambda* ulambda = lambda->uses().front().def()->as_lambda();
        Lambda* dropped = scope.drop(ulambda->args());
        ulambda->jump(dropped->to(), dropped->args());
        dropped->destroy_body();
        lambda->destroy_body();
    }
}

} // namespace anydsl2
