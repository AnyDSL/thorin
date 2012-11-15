#include "anydsl2/analyses/scope.h"

#include "anydsl2/lambda.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

static void jump_to_param_users(LambdaSet& scope, Lambda* lambda);
static void walk_up(LambdaSet& scope, Lambda* lambda);
static void find_user(LambdaSet& scope, const Def* def);

LambdaSet find_scope(Lambda* lambda) {
    LambdaSet scope;
    scope.insert(lambda);
    jump_to_param_users(scope, lambda);

    return scope;
}

static void jump_to_param_users(LambdaSet& scope, Lambda* lambda) {
    for_all (param, lambda->params())
        find_user(scope, param);
}

static void find_user(LambdaSet& scope, const Def* def) {
    if (Lambda* lambda = def->isa_lambda())
        walk_up(scope, lambda);
    else {
        for_all (use, def->uses())
            find_user(scope, use.def());
    }
}

static void walk_up(LambdaSet& scope, Lambda* lambda) {
    if (scope.find(lambda) != scope.end())
        return;// already inside scope so break

    scope.insert(lambda);
    jump_to_param_users(scope, lambda);

    for_all (pred, lambda->preds())
        walk_up(scope, pred);
}

size_t Scope::number(const LambdaSet& lambdas, Lambda* cur, size_t i) {
    // mark as visited
    cur->sid_ = 0;

    // for each successor in scope
    for_all (succ, cur->succs()) {
        if (lambdas.find(succ) != lambdas.end() && succ->sid_invalid())
            i = number(lambdas, succ, i);
    }

    return (cur->sid_ = i) + 1;
}

Scope::Scope(Lambda* entry) {
    lambdas_ = find_scope(entry);

    for_all (lambda, lambdas_)
        lambda->invalidate_sid();

    size_t num = number(lambdas_, entry, 0);

    rpo_.alloc(num);
    preds_.alloc(num);
    succs_.alloc(num);

    for (LambdaSet::iterator i = lambdas_.begin(); i != lambdas_.end();) {
        LambdaSet::iterator j = i++;
        Lambda* lambda = *j;
        if (lambda->sid() >= num) { // lambda is unreachable
            lambdas_.erase(j);
            continue; 
        }

        size_t sid = num - 1 - lambda->sid_;
        lambda->sid_ = sid;
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

    assert(rpo_[0] == entry && "bug in numbering");
}

const Scope::Lambdas& Scope::preds(Lambda* lambda) const {
    assert(contains(lambda)); 
    return preds_[lambda->sid()]; 
}

const Scope::Lambdas& Scope::succs(Lambda* lambda) const {
    assert(contains(lambda)); 
    return succs_[lambda->sid()]; 
}


void Scope::reassign_sids() {
    for (size_t i = 0, e = rpo().size(); i != e; ++i)
        rpo_[i]->sid_ = i;
}

} // namespace anydsl2
