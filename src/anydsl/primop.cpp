#include "anydsl/primop.h"

#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"

namespace anydsl {

RelOp::RelOp(RelOpKind kind, const Def* ldef, const Def* rdef)
    : BinOp((IndexKind) kind, ldef->world().type_u1(), ldef, rdef)
{}

Select::Select(const Def* cond, const Def* t, const Def* f) 
    : PrimOp(Index_Select, t->type(), 3)
{
    setOp(0, cond);
    setOp(1, t);
    setOp(2, f);
    anydsl_assert(cond->type() == world().type_u1(), "condition must be of u1 type");
    anydsl_assert(t->type() == f->type(), "types of both values must be equal");
}

Extract::Extract(const Def* tuple, const PrimLit* elem) 
    : PrimOp(Index_Extract, tuple->type()->as<Sigma>()->get(elem), 2)
{
    setOp(0, tuple);
    setOp(1, elem);
}
    
Insert::Insert(const Def* tuple, const PrimLit* elem, const Def* value)
    : PrimOp(Index_Insert, tuple->type(), 3)
{
    setOp(0, tuple);
    setOp(1, elem);
    setOp(2, value);
    anydsl_assert(tuple->type()->as<Sigma>()->get(elem) == value->type(), "type error");
}

Tuple::Tuple(World& world, const Def* const* begin, const Def* const* end) 
    : PrimOp(Index_Tuple, 0, std::distance(begin, end))
{
    if (numOps() == 0) {
        setType(world.sigma0());
    } else {
        const Type** types = new const Type*[std::distance(begin, end)];
        size_t x = 0;
        for (const Def* const* i = begin; i != end; ++i, ++x) {
            setOp(x, *i);
            types[x] = (*i)->type();
        }

        setType(world.sigma(types, types + numOps()));
        delete[] types;
    }
}

} // namespace anydsl
