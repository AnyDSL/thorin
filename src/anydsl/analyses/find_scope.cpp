#include "anydsl/analyses/find_scope.h"

#include "anydsl/lambda.h"

#include "anydsl/util/for_all.h"

namespace anydsl {

static void jump_to_param_users(LambdaSet& scope, const Lambda* lambda);
static void walk_up(LambdaSet& scope, const Lambda* lambda);
static void find_user(LambdaSet& scope, const Def* def);

LambdaSet find_scope(const Lambda* lambda) {
    LambdaSet scope;
    scope.insert(lambda);
    jump_to_param_users(scope, lambda);

    return scope;
}

static void jump_to_param_users(LambdaSet& scope, const Lambda* lambda) {
    for_all (param, lambda->params())
        find_user(scope, param);
}

static void find_user(LambdaSet& scope, const Def* def) {
    if (const Lambda* lambda = def->isa<Lambda>())
        walk_up(scope, lambda);
    else
        for_all (use, def->uses())
            find_user(scope, use.def());
}

static void walk_up(LambdaSet& scope, const Lambda* lambda) {
    if (scope.find(lambda) != scope.end())
        return;// already inside scope so break

    scope.insert(lambda);
    jump_to_param_users(scope, lambda);

    for_all (caller, lambda->callers())
        walk_up(scope, caller);
}

} // namespace anydsl
