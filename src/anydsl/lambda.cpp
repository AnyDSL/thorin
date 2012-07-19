#include "anydsl/lambda.h"

#include <boost/scoped_array.hpp>

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi, 0)
{
    for (size_t i = 0, e = pi->numOps(); i != e; ++i)
        world().createParam(pi->get(i), this, i);
}

const Pi* Lambda::pi() const {
    return scast<Pi>(type());
}

const Param* Lambda::appendParam(const Type* type) {
    size_t size = pi()->numOps();

    boost::scoped_array<const Type*> types(new const Type*[size + 1]);

    for (size_t i = 0; i < size; ++i)
        types[i] = pi()->get(i);

    types[size] = type;

    setType(world().pi(types.get(), types.get() + size + 1));

    return world().createParam(type, this, size);
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


bool Lambda::equal(const Def* other) const {
    return other->isa<Lambda>() && this == other->as<Lambda>();
}

size_t Lambda::hash() const {
    return boost::hash_value(this);
}

} // namespace anydsl
