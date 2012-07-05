#include "anydsl/lambda.h"

#include "anydsl/jump.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi, 1)
    , final_(false)
    , numArgs_(pi->numOps())
{
    for (size_t i = 0, e = pi->numOps(); i != e; ++i)
        new Param(pi->get(i), this, i);
}

Lambda::Lambda()
    : Def(Index_Lambda, 0, 1)
    , final_(false)
    , numArgs_(0)
{}

const Pi* Lambda::pi() const {
    return scast<Pi>(type());
}

void Lambda::setJump(const Jump* j) { 
    anydsl_assert(!ops_[0], "jump already set");
    setOp(0, j); 
}

const Param* Lambda::appendParam(const Type* type) {
    assert(!final_);
    anydsl_assert(!this->type(), "type already set -- you are not allowed to add any more params");

    return new Param(type, this, numArgs_++);
}

void Lambda::calcType(World& world) {
    anydsl_assert(!type(), "type already set");
    std::vector<const Type*> types;

    for_all (param, params())
        types.push_back(param->type());

    setType(world.pi(types.begin().base(), types.end().base()));
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
