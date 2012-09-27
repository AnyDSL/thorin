#ifndef ANYDSL_ANALYSES_SCOPE_H
#define ANYDSL_ANALYSES_SCOPE_H

#include <boost/unordered_set.hpp>

#include "anydsl/util/array.h"

namespace anydsl {

class Lambda;

typedef boost::unordered_set<Lambda*> LambdaSet;

LambdaSet find_scope(Lambda* entry);

class Scope {
public:

    typedef Array<Lambda*> Lambdas;

    explicit Scope(Lambda* entry);

    bool contains(Lambda* lambda) const { return lambdas_.find(lambda) != lambdas_.end(); }
    const LambdaSet& lambdas() const { return lambdas_; }
    const Lambdas& rpo() const { return rpo_; }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    const Lambdas& preds(Lambda* lambda) const;
    const Lambdas& succs(Lambda* lambda) const;

    Lambda* entry() const { return rpo_[0]; }
    size_t size() const { return lambdas_.size(); }

private:

    LambdaSet lambdas_;
    Lambdas rpo_;
    Array<Lambdas> preds_;
    Array<Lambdas> succs_;
};

} // namespace anydsl

#endif

