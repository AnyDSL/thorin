#include "anydsl/lambda.h"

#include <algorithm>

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi, uint32_t flags)
    : Def(Node_Lambda, pi, 0)
    , flags_(flags)
{
    params_.reserve(pi->size());

    size_t i = 0;
    for_all (elem, pi->elems())
        params_.insert(world().param(elem, this, i++));
}

Lambda::~Lambda() {
    for_all (param, params())
        param->lambda_ = 0;

}

static void findLambdas(const Def* def, LambdaSet& result) {
    if (const Lambda* lambda = def->isa<Lambda>()) {
        result.insert(lambda);
        return;
    }

    for_all (op, def->ops())
        findLambdas(op, result);
}

LambdaSet Lambda::targets() const {
    LambdaSet result;
    findLambdas(to(), result);

    return result;
}

LambdaSet Lambda::succ() const {
    LambdaSet result;

    for_all (def, ops())
        findLambdas(def, result);

    return result;
}

static void findCallers(const Def* def, LambdaSet& result) {
    if (const Lambda* lambda = def->isa<Lambda>()) {
        result.insert(lambda);
        return;
    }

    anydsl_assert(def->isa<PrimOp>(), "not a PrimOp");

    for_all (use, def->uses())
        findCallers(use.def(), result);
}

LambdaSet Lambda::callers() const {
    LambdaSet result;

    for_all (use, uses())
        findCallers(use.def(), result);

    return result;
}

const Pi* Lambda::pi() const {
    return scast<Pi>(type());
}

const Param* Lambda::appendParam(const Type* type) {
    size_t size = pi()->elems().size();

    Array<const Type*> elems(size + 1);
    *std::copy(pi()->elems().begin(), pi()->elems().end(), elems.begin()) = type;
    setType(world().pi(elems));

    const Param* param = world().param(type, this, size);
    params_.insert(param);

    return param;
}

bool Lambda::equal(const Def* other) const {
    return this == other;
}

size_t Lambda::hash() const {
    return boost::hash_value(this);
}

const Param* Lambda::param(size_t i) const {
    Param p(0, 0, i);
    return *params_.find(&p);
}

Params::const_iterator Lambda::ho_begin() const {
    Params::const_iterator result = params_.begin();

    while (result != params_.end() && !(*result)->type()->isa<Pi>())
        ++result;

    return result;
}

void Lambda::ho_next(Params::const_iterator& i) const {
    ++i;

    while (i != params_.end() && !(*i)->type()->isa<Pi>())
        ++i;
}

} // namespace anydsl
