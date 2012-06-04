#include "anydsl/primop.h"

#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"

namespace anydsl {

RelOp::RelOp(RelOpKind kind, Def* ldef, Def* rdef)
    : BinOp((IndexKind) kind, ldef->world().type_u1(), ldef, rdef)
{}

Select::Select(Def* cond, Def* t, Def* f) 
    : PrimOp(Index_Select, t->type(), 3)
{
    setOp(0, cond);
    setOp(1, t);
    setOp(2, f);
    anydsl_assert(cond->type() == world().type_u1(), "condition must be of u1 type");
    anydsl_assert(t->type() == f->type(), "types of both values must be equal");
}

Proj::Proj(Def* tuple, PrimLit* elem) 
    : PrimOp(Index_Proj, tuple->type()->as<Sigma>()->get(elem), 2)
{
    setOp(0, tuple);
    setOp(1, elem);
}
    
Insert::Insert(Def* tuple, PrimLit* elem, Def* value)
    : PrimOp(Index_Insert, tuple->type(), 3)
{
    setOp(0, tuple);
    setOp(1, elem);
    setOp(2, value);
    anydsl_assert(tuple->type()->as<Sigma>()->get(elem) == value->type(), "type error");
}

} // namespace anydsl
