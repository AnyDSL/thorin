#include "anydsl/lambda.h"

#include <algorithm>

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(size_t gid, const Pi* pi, uint32_t flags)
    : Def(Node_Lambda, pi)
    , gid_(gid)
    , flags_(flags)
{
    params_.reserve(pi->size());
}

Lambda* Lambda::stub() const { 
    Lambda* result = world().lambda(pi(), flags());
    result->debug = debug;

    for (size_t i = 0, e = params().size(); i != e; ++i)
        result->param(i)->debug = param(i)->debug;

    return result;
}

const Pi* Lambda::pi() const { return type()->as<Pi>(); }
const Pi* Lambda::to_pi() const { return to()->type()->as<Pi>(); }

const Param* Lambda::append_param(const Type* type) {
    size_t size = pi()->elems().size();

    Array<const Type*> elems(size + 1);
    *std::copy(pi()->elems().begin(), pi()->elems().end(), elems.begin()) = type;
    set_type(world().pi(elems));

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

void Lambda::close() {
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

void Lambda::reclose() {
    adjacencies_.~Array();
    new (&adjacencies_) Array<const Lambda*>();
    close();
}

template<bool fo>
Array<const Param*> Lambda::classify_params() const {
    Array<const Param*> res(params().size());

    size_t size = 0;
    for_all (param, params())
        if (fo ^ (param->type()->isa<Pi>() != 0))
            res[size++] = param;

    res.shrink(size);

    return res;
}

template<bool fo>
Array<const Def*> Lambda::classify_args() const {
    Array<const Def*> res(args().size());

    size_t size = 0;
    for_all (arg, args())
        if (fo ^ (arg->type()->isa<Pi>() != 0))
            res[size++] = arg;

    res.shrink(size);

    return res;
}

bool Lambda::is_cascading() const {
    if (uses().size() != 1)
        return false;

    Use use = *uses().begin();
    return !use.def()->isa<Lambda>() || !use.index() > 0;
}

Array<const Param*> Lambda::fo_params() const { return classify_params<true>(); }
Array<const Param*> Lambda::ho_params() const { return classify_params<false>(); }
Array<const Def*> Lambda::fo_args() const { return classify_args<true>(); }
Array<const Def*> Lambda::ho_args() const { return classify_args<false>(); }
bool Lambda::is_fo()  const { return pi()->is_fo(); }
bool Lambda::is_ho() const { return pi()->is_ho(); }

void Lambda::jump(const Def* to, ArrayRef<const Def*> args) {
    alloc(args.size() + 1);
    set_op(0, to);

    size_t x = 1;
    for_all (arg, args)
        set_op(x++, arg);

    close();
}

void Lambda::branch(const Def* cond, const Def* tto, const Def*  fto) {
    return jump(world().select(cond, tto, fto), ArrayRef<const Def*>(0, 0));
}


} // namespace anydsl
