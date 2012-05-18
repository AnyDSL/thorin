#include "anydsl/air/sigmaop.h"

#include "anydsl/air/type.h"
#include "anydsl/support/hash.h"

namespace anydsl {

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
