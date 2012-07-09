#include "anydsl/lambda.h"

#include <boost/scoped_array.hpp>

#include "anydsl/type.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi, 0)
    , final_(false)
    , numArgs_(pi->numOps())
{
    for (size_t i = 0, e = pi->numOps(); i != e; ++i)
        new Param(pi->get(i), this, i);
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

    return new Param(type, this, numArgs_++);
}

void Lambda::calcType(World& world) {
    anydsl_assert(!type(), "type already set");
    size_t size = unordered_params().size();
    boost::scoped_array<const Type*> types(new const Type*[size]);

    for_all (param, unordered_params())
        types[param->index()] = param->type();

    setType(world.pi(types.get(), types.get() + size));;
}

std::vector<const Lambda*> Lambda::succ() const {
    std::vector<const Lambda*> result;

    for_all (def, ops()) {
        if (const Lambda* lambda = def->isa<Lambda>()) {
            result.push_back(lambda);
        } else if (const Select* select = todef()->isa<Select>()) {
            const Lambda* tlambda = select->tdef()->as<Lambda>();
            const Lambda* flambda = select->fdef()->as<Lambda>();
            result.push_back(tlambda);
            result.push_back(flambda);
        }
    }

    return result;
}

bool Lambda::equal(const Def* other) const {
    return other->isa<Lambda>() && this == other->as<Lambda>();
}

size_t Lambda::hash() const {
    return boost::hash_value(this);
}

Params Lambda::params() const { 
    size_t size = unordered_params().size();
    Params result(size);

    for_all (param, unordered_params())
        result[param->index()] = param;

    return result;
}

size_t Lambda::numParams() const {
    assert(type());
    return pi()->numOps();
}

} // namespace anydsl
