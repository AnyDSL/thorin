#ifndef ANYDSL2_ANALYSES_SCOPE_H
#define ANYDSL2_ANALYSES_SCOPE_H

#include <boost/unordered_set.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/util/array.h"

namespace anydsl2 {

class Lambda;

LambdaSet find_scope(Lambda* entry);

class Scope {
public:

    typedef Array<Lambda*> Lambdas;
    typedef std::vector<const Def*> FreeVariables;

    explicit Scope(Lambda* entry);

    bool contains(Lambda* lambda) const { return lambdas_.find(lambda) != lambdas_.end(); }
    const LambdaSet& lambdas() const { return lambdas_; }
    const Lambdas& rpo() const { return rpo_; }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    const Lambdas& preds(Lambda* lambda) const;
    const Lambdas& succs(Lambda* lambda) const;
    Lambda* entry() const { return rpo_[0]; }
    size_t size() const { return lambdas_.size(); }
    void reassign_sids();
    World& world() const { return entry()->world(); }
    FreeVariables free_variables() const;

private:

    static size_t number(const LambdaSet& lambdas, Lambda* cur, size_t i);

    LambdaSet lambdas_;
    Lambdas rpo_;
    Array<Lambdas> preds_;
    Array<Lambdas> succs_;
};

} // namespace anydsl2

#endif

