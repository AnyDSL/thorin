#include "anydsl/lambda.h"

#include <boost/scoped_array.hpp>

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi, Params& params)
    : Def(Index_Lambda, pi, 0)
    , final_(false)
    , numArgs_(pi->numOps())
{
    for (size_t i = 0, e = pi->numOps(); i != e; ++i)
        params.push_back(world().createParam(pi->get(i), this, i));
}

Lambda::Lambda()
    : Def(Index_Lambda, 0, 0)
    , final_(false)
    , numArgs_(0)
{}

const Pi* Lambda::pi() const {
    return scast<Pi>(type());
}

void Lambda::jumps(const Def* to, const Def* const* begin, const Def* const* end) { 
    alloc(std::distance(begin, end) + 1);

    setOp(0, to);

    const Def* const* i = begin;
    for (size_t x = 1; i != end; ++x, ++i)
        setOp(x, *i);
}

void Lambda::branches(const Def* cond, const Def* tto, const Def*  fto) {
    return jumps(cond->world().createSelect(cond, tto, fto), 0, 0);
}

const Param* Lambda::appendParam(const Type* type) {
    assert(!final_);
    anydsl_assert(!this->type(), "type already set -- you are not allowed to add any more params");

    return type->world().createParam(type, this, numArgs_++);
}

void Lambda::calcType(World& world, const Params& params) {
    anydsl_assert(!type(), "type already set");
    size_t size = params.size();
    boost::scoped_array<const Type*> types(new const Type*[size]);

    for (size_t i = 0; i < size; ++i)
        types[i] = params[i]->type();

    setType(world.pi(types.get(), types.get() + size));;
}

static void findLambdas(const Def* def, LambdaSet& result) {
    if (const Lambda* lambda = def->isa<Lambda>()) {
        result.insert(lambda);
        return;
    }

    for_all (op, def->ops())
        findLambdas(op, result);
}

LambdaSet Lambda::to() const {
    LambdaSet result;
    findLambdas(todef(), result);

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

bool Lambda::equal(const Def* other) const {
    return other->isa<Lambda>() && this == other->as<Lambda>();
}

size_t Lambda::hash() const {
    return boost::hash_value(this);
}

static void findParam(const Def* def, const Lambda* lambda, Params& params) { 
    if (const Param* param = def->isa<Param>()) {
        if (param->lambda() == lambda)
            params.push_back(param);

        return;
    } else if (def->isa<Lambda>())
        return;

    for_all (op, def->ops())
        findParam(op, lambda, params);
}

Params Lambda::params() const { 
    return world().findParams(this);
}

} // namespace anydsl
