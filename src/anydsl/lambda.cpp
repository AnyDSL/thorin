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
    , adjacencies_(0)
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

const Pi* Lambda::pi() const {
    return type()->as<Pi>();
}

const Pi* Lambda::to_pi() const {
    return to()->type()->as<Pi>();
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

Array<const Param*> Lambda::copy_params() const {
    Array<const Param*> result(pi()->size());

    size_t i = 0;
    for_all (param, params()) {
        result[i] = param->index() == i ? param : 0;
        ++i;
    }

    return result;
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

Params::const_iterator Lambda::fo_begin() const {
    Params::const_iterator result = params_.begin();

    while (result != params_.end() && (*result)->type()->isa<Pi>())
        ++result;

    return result;
}

void Lambda::fo_next(Params::const_iterator& i) const {
    ++i;

    while (i != params_.end() && (*i)->type()->isa<Pi>())
        ++i;
}

void Lambda::shrink(ArrayRef<size_t> drop) {
    assert(!drop.empty());

    // r -> read, w -> write, d -> drop
    for (size_t r = drop.front(), w = drop.front(), d = 0, e = args().size(); r != e; ++r) {
        op(r + 1)->unregisterUse(r + 1, this);

        if (d < drop.size() && drop[d] == r)
            ++d;
        else
            setOp(w + 1, op(r + 1));
    }

    Def::shrink(size() - drop.size());
}

static void find_lambdas(const Def* def, LambdaSet& result) {
    if (const Lambda* lambda = def->isa<Lambda>()) {
        result.insert(lambda);
        return;
    }

    for_all (op, def->ops())
        find_lambdas(op, result);
}

static void find_preds(const Def* def, LambdaSet& result) {
    if (const Lambda* lambda = def->isa<Lambda>()) {
        result.insert(lambda);
        return;
    }

    anydsl_assert(def->isa<PrimOp>(), "not a PrimOp");

    for_all (use, def->uses())
        find_preds(use.def(), result);
}

LambdaSet Lambda::preds() const {
    LambdaSet result;

    for_all (use, uses())
        find_preds(use.def(), result);

    return result;
}

void Lambda::close() {
    LambdaSet targets, hos;

    find_lambdas(to(), targets);

    for_all (def, args())
        find_lambdas(def, hos);

    adjacencies_.~Array();
    new (&adjacencies_) Array<const Lambda*>(targets.size() + hos.size());

    size_t i = 0;
    for_all (target, targets)
        adjacencies_[i++] = target;

    hosBegin_ = i;
    for_all (ho, hos)
        adjacencies_[i++] = ho;

}

} // namespace anydsl
