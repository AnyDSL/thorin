#include "anydsl/air/lambda.h"

#include "anydsl/air/terminator.h"
#include "anydsl/air/type.h"
#include "anydsl/air/world.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

Lambda::Lambda(const Pi* pi)
    : Def(Index_Lambda, pi)
    , parent_(0)
    , terminator_(0)
{}

Lambda::Lambda(World& world)
    : Def(Index_Lambda, world.pi0())
    , parent_(0)
    , terminator_(0)
{}

Lambda::~Lambda() {
    delete terminator_;

    FOREACH(param, params_) 
        delete param;
}

void Lambda::insert(Lambda* lambda) {
    anydsl_assert(lambda, "lambda invalid");
    anydsl_assert(lambda->parent_ == 0, "already has a parent");
    anydsl_assert(fix_.find(lambda) == fix_.end(), "already innserted");
    fix_.insert(lambda);
    lambda->parent_ = this;
}

void Lambda::remove(Lambda* lambda) {
    anydsl_assert(lambda, "lambda invalid");
    anydsl_assert(lambda->parent_, "parent must be set");
    anydsl_assert(fix_.find(lambda) != fix_.end(), "lambda not inside fix");
    fix_.erase(lambda);
    lambda->parent_ = 0;
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

int Lambda::depth() {
    int res = 0;

    Lambda* i = parent_;

    while (i) {
        ++res;
        i = i->parent_;
    }

    return res;
}

} // namespace anydsl
