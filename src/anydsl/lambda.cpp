#include "anydsl/lambda.h"

#include <algorithm>

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi, uint32_t flags)
    : Def(Node_Lambda, pi)
    , flags_(flags)
{
    params_.reserve(pi->size());

    size_t i = 0;
    for_all (elem, pi->elems())
        params_.push_back(world().param(elem, this, i++));
}

const Pi* Lambda::pi() const { return type()->as<Pi>(); }
const Pi* Lambda::to_pi() const { return to()->type()->as<Pi>(); }

const Param* Lambda::append_param(const Type* type) {
    size_t size = pi()->elems().size();

    Array<const Type*> elems(size + 1);
    *std::copy(pi()->elems().begin(), pi()->elems().end(), elems.begin()) = type;
    setType(world().pi(elems));

    const Param* param = world().param(type, this, size);
    params_.push_back(param);

    return param;
}

bool Lambda::equal(const Def* other) const { return this == other; }
size_t Lambda::hash() const { return boost::hash_value(this); }

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

void Lambda::close(size_t gid) {
    gid_ = gid;

    LambdaSet targets, hos;

    find_lambdas(to(), targets);

    for_all (def, args())
        find_lambdas(def, hos);

    adjacencies_.alloc(targets.size() + hos.size());

    size_t i = 0;
    for_all (target, targets)
        adjacencies_[i++] = target;

    hos_begin_ = i;
    for_all (ho, hos)
        adjacencies_[i++] = ho;

    anydsl_assert(pi()->size() == params().size(), "type does not honor size of params");
}

template<bool first_order>
Array<const Param*> Lambda::classify_params() const {
    Array<const Param*> res(params().size());

    size_t size = 0;
    for_all (param, params())
        if (first_order ^ (param->type()->isa<Pi>() != 0))
            res[size++] = param;

    res.shrink(size);

    return res;
}

template<bool first_order>
Array<const Def*> Lambda::classify_args() const {
    Array<const Def*> res(args().size());

    size_t size = 0;
    for_all (arg, args())
        if (first_order ^ (arg->type()->isa<Pi>() != 0))
            res[size++] = arg;

    res.shrink(size);

    return res;
}

Array<const Param*> Lambda::first_order_params() const { return classify_params<true>(); }
Array<const Param*> Lambda::higher_order_params() const { return classify_params<false>(); }
Array<const Def*> Lambda::first_order_args() const { return classify_args<true>(); }
Array<const Def*> Lambda::higher_order_args() const { return classify_args<false>(); }
bool Lambda::is_first_order()  const { return pi()->is_first_order(); }
bool Lambda::is_higher_order() const { return pi()->is_higher_order(); }

} // namespace anydsl
