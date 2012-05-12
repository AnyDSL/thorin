#include "anydsl/air/constant.h"
#include "anydsl/air/type.h"

#include "anydsl/util/foreach.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimConst::PrimConst(PrimTypeKind kind, Box box, const std::string& debug)
    : Constant((IndexKind) kind, universe().getPrimType(kind), debug)
    , box_(box)
{}

uint64_t PrimConst::hash() const {
    anydsl_assert(sizeof(Box) == 8, "Box has unexpected size");
    return (uint64_t(index()) << 32) | bcast<uint64_t, Box>((box()));
}

//------------------------------------------------------------------------------


uint64_t Tuple::hash() const {
    // TODO
    return 0;
}

//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------

} // namespace anydsl
