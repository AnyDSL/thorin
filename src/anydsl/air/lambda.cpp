#include "anydsl/air/lambda.h"

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
    : Def(Index_Lambda, world.emptyPi())
    , parent_(parent)
    , terminator_(0)
{}

Lambda::~Lambda() {
    FOREACH(lambda, fix_)
        delete lambda;
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

} // namespace anydsl
