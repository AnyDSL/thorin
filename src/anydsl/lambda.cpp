#include "anydsl/lambda.h"

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi, 0)
{
    for (size_t i = 0, e = pi->numOps(); i != e; ++i)
        world().param(pi->get(i), this, i);
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

Params Lambda::params() const { 
    return world().findParams(this);
}

const Pi* Lambda::pi() const {
    return scast<Pi>(type());
}

const Param* Lambda::appendParam(const Type* type) {
    size_t size = pi()->numOps();

    Array<const Type*> elems(size + 1);

    for (size_t i = 0; i < size; ++i)
        elems[i] = pi()->get(i);

    elems[size] = type;

    setType(world().pi(elems));

    return world().param(type, this, size);
}


bool Lambda::equal(const Def* other) const {
    return this == other;
}

size_t Lambda::hash() const {
    return boost::hash_value(this);
}

} // namespace anydsl
