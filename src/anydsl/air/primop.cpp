#include "anydsl/air/primop.h"

#include "anydsl/air/type.h"
#include "anydsl/air/world.h"

namespace anydsl {

//------------------------------------------------------------------------------

RelOp::RelOp(RelOpKind kind, Def* ldef, Def* rdef)
    : BinOp((IndexKind) kind, ldef->world().type_u1(), ldef, rdef)
{
    anydsl_assert(ldef->type() == rdef->type(), "type are not equal");
}

//------------------------------------------------------------------------------

SigmaOp::SigmaOp(IndexKind index, const Type* type, Def* tuple, PrimLit* elem)
    : PrimOp(index, type)
    , tuple(tuple, this)
    , elem_(elem)
{
    anydsl_assert(tuple->as<Sigma>(), "must be of Sigma type");
}

//------------------------------------------------------------------------------

Extract::Extract(Def* tuple, PrimLit* elem)
    : SigmaOp(Index_Extract, scast<Sigma>(tuple->type())->get(elem), tuple, elem)
{}

//------------------------------------------------------------------------------

} // namespace anydsl
