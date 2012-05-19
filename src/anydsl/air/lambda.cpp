#include "anydsl/air/lambda.h"

#include "anydsl/air/terminator.h"
#include "anydsl/air/type.h"
#include "anydsl/air/world.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

Lambda::Lambda(Lambda* parent, const Type* type)
    : Def(Index_Lambda, type)
    , parent_(parent)
    , terminator_(0)
{
    anydsl_assert(type->isa<Pi>(), "type must be a Pi");
}

Lambda::Lambda(World& world, Lambda* parent)
    : Def(Index_Lambda, world.pi())
    , parent_(parent)
    , terminator_(0)
{}

Lambda::~Lambda() {
    std::cout << "fjdkfjdlk" << std::endl;
    delete terminator_;
}

void Lambda::insert(Lambda* lambda) {
    anydsl_assert(lambda, "lambda invalid");
    anydsl_assert(fix_.find(lambda) == fix_.end(), "already innserted");
    fix_.insert(lambda);
}

void Lambda::remove(Lambda* lambda) {
    anydsl_assert(lambda, "lambda invalid");
    anydsl_assert(fix_.find(lambda) != fix_.end(), "lambda not inside fix");
    fix_.erase(lambda);
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
