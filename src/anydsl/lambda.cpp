#include "anydsl/lambda.h"

#include "anydsl/jump.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi, 1)
    , final_(true)
    , params_(pi->numOps())
{
    for_all (&param, params_)
        param = new Param(this);
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

Param* Lambda::appendParam() {
    assert(!final_);

    Param* param = new Param(this);
    params_.push_back(param);

    return param;
}

} // namespace anydsl
