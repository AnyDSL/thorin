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
    , tuple(*ops_append((Def*) vn.op1))
    , elem (*ops_append((Def*) vn.op2))
{
    anydsl_assert(vn.index == Index_Proj, "wrong index in VN");
}

//------------------------------------------------------------------------------

Select::Select(const ValueNumber& vn) 
    : PrimOp(Index_Select, ((Def*) vn.op2)->type())
    , cond(*ops_append((Def*) vn.op1))
    , tuse(*ops_append((Def*) vn.op2))
    , fuse(*ops_append((Def*) vn.op3))
{
    anydsl_assert(cond.def()->type() == world().type_u1(), "condition must be of u1 type");
    anydsl_assert(tuse.def()->type() == fuse.def()->type(), "types of both values must be equal");
}

//------------------------------------------------------------------------------

} // namespace anydsl
