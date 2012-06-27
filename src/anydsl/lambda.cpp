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

void Lambda::setJump(const Jump* jump) { 
    setOp(0, jump); 
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
        types.push_back(param.def()->type());

    setType(world.pi(types.begin().base(), types.end().base()));

    std::cout << "in calcType" << std::endl;
    type()->dump();
    std::cout << std::endl;
}

bool Lambda::equal(const Def* other) const {
    return false;
}

size_t Lambda::hash() const {
    return boost::hash_value(this);
}

} // namespace anydsl
