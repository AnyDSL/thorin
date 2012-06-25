#include "anydsl/lambda.h"

#include "anydsl/jump.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Value(Index_Lambda, pi, 1)
    , final_(false)
    , params_(pi->numOps())
{
    size_t i = 0;
    for_all (&param, params_) {
        param = new Param(this, i, pi->get(i));
        ++i;
    }
}

Lambda::Lambda()
    : Value(Index_Lambda, 0, 1)
    , final_(false)
{}

const Pi* Lambda::pi() const {
    return scast<Pi>(type());
}

void Lambda::setJump(const Jump* jump) { 
    setOp(0, jump); 
}

Param* Lambda::appendParam(const Type* type) {
    assert(!final_);
    anydsl_assert(!this->type(), "type already set -- you are not allowed to add any more params");

    Param* param = new Param(this, params_.size(), type);
    params_.push_back(param);

    return param;
}

void Lambda::calcType(World& world) {
    anydsl_assert(!type(), "type already set");
    std::vector<const Type*> types;

    for_all (param, params())
        types.push_back(param->type());

    setType(world.pi(types.begin().base(), types.end().base()));
}

} // namespace anydsl
