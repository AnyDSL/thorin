#include "anydsl/lambda.h"

#include "anydsl/jump.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi, 1)
{}

Lambda::Lambda(World& world)
    : Def(Index_Lambda, world.pi0(), 1)
{}

Lambda::~Lambda() {
    delete jump_;

#ifndef NDEBUG
    FOREACH(param, params_)
        anydsl_assert(param->uses().empty(), "there are still uses pointing to param '") 
            << param->debug << "'";
#endif

    FOREACH(param, params_) 
        delete param;
}

ParamIter Lambda::appendParam(const Type* type) {
    // get copy of parameter type list
    Types types = pi()->types();

    // append new param type
    types.push_back(type);

    // update pi type
    setType(world().pi(types));

    // create and register new param
    ParamIter i = params_.insert(params_.end(), new Param(this, type));

    return i;
}

const Pi* Lambda::pi() const {
    return scast<Pi>(type());
}

} // namespace anydsl
