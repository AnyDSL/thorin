#include "anydsl/air/primop.h"

#include "anydsl/air/type.h"
#include "anydsl/air/world.h"

namespace anydsl {

RelOp::RelOp(const ValueNumber& vn)
    : BinOp((IndexKind) vn.index, 
            ((Def*) vn.op1)->type()->world().type_u1(), 
            (Def*) vn.op1, 
            (Def*) vn.op2)
{
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
