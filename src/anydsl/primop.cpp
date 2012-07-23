#include "anydsl/primop.h"

#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"

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

TupleOp::TupleOp(IndexKind indexKind, const Type* type, size_t numOps, const Def* tuple, uint32_t index)
    : PrimOp(indexKind, type, numOps)
    , index_(index)
{
    setOp(0, tuple);
}

bool TupleOp::equal(const Def* other) const {
    if (!Def::equal(other))
        return false;

    return index() == other->as<TupleOp>()->index();
}

size_t TupleOp::hash() const {
    size_t seed = Def::hash();
    boost::hash_combine(seed, index());

    return seed;
}

Extract::Extract(const Def* tuple, uint32_t index)
    : TupleOp(Index_Extract, tuple->type()->as<Sigma>()->get(index), 1, tuple, index)
{
    setOp(0, tuple);
}

Insert::Insert(const Def* tuple, uint32_t index, const Def* value)
    : TupleOp(Index_Insert, tuple->type(), 2, tuple, index)
{
    setOp(1, value);
    anydsl_assert(tuple->type()->as<Sigma>()->get(index) == value->type(), "type error");
}

Tuple::Tuple(World& world, ArrayRef<const Def*> args) 
    : PrimOp(Index_Tuple, 0, args.size())
{
    if (numOps() == 0) {
        setType(world.sigma0());
    } else {
        Array<const Type*> types(numOps());
        size_t x = 0;
        for_all (arg, args)
            types[x++] = arg->type();

        setType(world.sigma(types));
    }
}

} // namespace anydsl
