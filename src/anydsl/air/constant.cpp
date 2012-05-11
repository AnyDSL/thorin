#include "anydsl/air/constant.h"

#include "anydsl/util/foreach.h"

namespace anydsl {

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
