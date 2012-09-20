#include "anydsl/analyses/scope.h"

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
    else {
        for_all (use, def->uses())
            find_user(scope, use.def());
    }
}

static void walk_up(LambdaSet& scope, const Lambda* lambda) {
    if (scope.find(lambda) != scope.end())
        return;// already inside scope so break

    scope.insert(lambda);
    jump_to_param_users(scope, lambda);

    for_all (pred, lambda->preds())
        walk_up(scope, pred);
}

static size_t number(const LambdaSet& lambdas, const Lambda* cur, size_t i) {
    // mark as visited
    cur->sid = 0;

    // for each successor in scope
    for_all (succ, cur->succs()) {
        if (lambdas.find(succ) != lambdas.end() && succ->sid_invalid())
            i = number(lambdas, succ, i);
    }

    cur->sid = i;

    return i - 1;
}

Scope::Scope(const Lambda* entry)
    : lambdas_(find_scope(entry))
    , rpo_(lambdas_.size())
    , preds_(lambdas_.size())
    , succs_(lambdas_.size())
{
    for_all (lambda, lambdas_)
        lambda->invalidate_sid();

    number(lambdas_, entry, size() - 1);

    for_all (lambda, lambdas_) {
        size_t sid = lambda->sid;
        rpo_[sid] = lambda;

        {
            Lambdas& succs = succs_[sid];
            succs.alloc(lambda->succs().size());
            size_t i = 0;
            for_all (succ, lambda->succs()) {
                if (lambdas_.find(succ) != lambdas_.end())
                    succs[i++] = succ;
            }
            succs.shrink(i);
        }
        {
            Lambdas& preds = preds_[sid];
            preds.alloc(lambda->preds().size());
            size_t i = 0;
            for_all (pred, lambda->preds()) {
                if (lambdas_.find(pred) != lambdas_.end())
                    preds[i++] = pred;
            }
            preds.shrink(i);
        }
    }
}

const Scope::Lambdas& Scope::preds(const Lambda* lambda) { return preds_[lambda->sid]; }
const Scope::Lambdas& Scope::succs(const Lambda* lambda) { return succs_[lambda->sid]; }

} // namespace anydsl
