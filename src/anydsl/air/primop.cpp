#include "anydsl/air/primop.h"

#include "anydsl/air/literal.h"
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

Proj::Proj(const ValueNumber& vn)
    : PrimOp(Index_Proj, 
             ((Def*)vn.op1)->type()->as<Sigma>()->get(((Def*) vn.op2)->as<PrimLit>()))
    , tuple(this, (Def*) vn.op1)
    , elem(this, (Def*) vn.op2)
{
    anydsl_assert(vn.index == Index_Proj, "wrong index in VN");
}

//------------------------------------------------------------------------------

} // namespace anydsl
