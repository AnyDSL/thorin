#include "anydsl/air/literal.h"
#include "anydsl/air/type.h"
#include "anydsl/air/world.h"

#include "anydsl/util/foreach.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimLit::PrimLit(World& world, PrimTypeKind kind, Box box)
    : Literal((IndexKind) kind, world.type(kind))
    , box_(box)
{}

uint64_t PrimLit::hash() const {
    anydsl_assert(sizeof(Box) == 8, "Box has unexpected size");
    return (uint64_t(index()) << 32) | bcast<uint64_t, Box>((box()));
}

//------------------------------------------------------------------------------


uint64_t Tuple::hash() const {
    // TODO
    return 0;
}

//------------------------------------------------------------------------------

Lambda::Lambda(Lambda* parent, const Type* type)
    : Literal(Index_Lambda, type)
    , parent_(parent)
    , terminator_(0)
{
    anydsl_assert(type->isa<Pi>(), "type must be a Pi");
}

Lambda::Lambda(World& world, Lambda* parent)
    : Literal(Index_Lambda, world.emptyPi())
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

//------------------------------------------------------------------------------

} // namespace anydsl
