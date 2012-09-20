#ifndef ANYDSL_ANALYSES_SCOPE_H
#define ANYDSL_ANALYSES_SCOPE_H

#include <boost/unordered_set.hpp>

#include "anydsl/util/array.h"

namespace anydsl {

class Lambda;

typedef boost::unordered_set<const Lambda*> LambdaSet;

LambdaSet find_scope(const Lambda* entry);

class Scope {
public:

    typedef Array<const Lambda*> Lambdas;

    Scope(const Lambda* entry);

    const LambdaSet& lambdas() const { return lambdas_; }
    const Lambdas& rpo() const { return rpo_; }
    const Lambdas& preds(const Lambda* lambda);
    const Lambdas& succs(const Lambda* lambda);

    const Lambda* entry() const { return rpo_[0]; }
    size_t size() const { return lambdas_.size(); }

private:

    LambdaSet lambdas_;
    Lambdas rpo_;
    Array<Lambdas> preds_;
    Array<Lambdas> succs_;
};

} // namespace anydsl

#endif

