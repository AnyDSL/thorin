#include "anydsl/lambda.h"

#include "anydsl/jump.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi, 1)
    , final_(false)
    , params_(pi->numOps())
{
    size_t i = 0;
    for_all (&param, params_)
        param = new Param(this, pi->get(i++));
}

Lambda::Lambda()
    : Def(Index_Lambda, 0, 1)
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

    Param* param = new Param(this, type);
    params_.push_back(param);

    return param;
}

} // namespace anydsl
