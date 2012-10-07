#include "anydsl/primop.h"

#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/world.h"
#include "anydsl/util/array.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

PrimOp* PrimOp::clone() const {
    PrimOp* result = stub();
    for (size_t i = 0, e = size(); i != e; ++i)
        result->set_op(i, op(i));

    return result;
}

//------------------------------------------------------------------------------

RelOp::RelOp(RelOpKind kind, const Def* lhs, const Def* rhs)
    : BinOp((NodeKind) kind, lhs->world().type_u1(), lhs, rhs)
{}

//------------------------------------------------------------------------------

bool ConvOp::equal(const Node* other) const {
    return Def::equal(other) && type() == other->as<ConvOp>()->type();
}

size_t ConvOp::hash() const {
    size_t seed = Def::hash();
    boost::hash_combine(seed, type());

    return seed;
}

//------------------------------------------------------------------------------

Select::Select(const Def* cond, const Def* t, const Def* f) 
    : PrimOp(Node_Select, 3, t->type())
{
    set_op(0, cond);
    set_op(1, t);
    set_op(2, f);
    anydsl_assert(cond->type() == world().type_u1(), "condition must be of u1 type");
    anydsl_assert(t->type() == f->type(), "types of both values must be equal");
}

//------------------------------------------------------------------------------

TupleOp::TupleOp(NodeKind kind, size_t size, const Type* type, const Def* tuple, u32 index)
    : PrimOp(kind, size, type)
    , index_(index)
{
    set_op(0, tuple);
}

bool TupleOp::equal(const Node* other) const {
    return Def::equal(other) && index() == other->as<TupleOp>()->index();
}

size_t TupleOp::hash() const {
    size_t seed = Def::hash();
    boost::hash_combine(seed, index());

    return seed;
}

//------------------------------------------------------------------------------

Extract::Extract(const Def* tuple, u32 index)
    : TupleOp(Node_Extract, 1, tuple->type()->as<Sigma>()->elem(index), tuple, index)
{}

//------------------------------------------------------------------------------

Insert::Insert(const Def* tuple, u32 index, const Def* value)
    : TupleOp(Node_Insert, 2, tuple->type(), tuple, index)
{
    set_op(1, value);
    anydsl_assert(tuple->type()->as<Sigma>()->elem(index) == value->type(), "type error");
}

//------------------------------------------------------------------------------

Tuple::Tuple(World& world, ArrayRef<const Def*> args) 
    : PrimOp(Node_Tuple, args.size(), (const Type*) 0)
{
    if (ops().empty()) {
        set_type(world.sigma0());
    } else {
        Array<const Type*> types(ops().size());
        size_t x = 0;
        for_all (arg, args) {
            set_op(x, args[x]);
            types[x++] = arg->type();
        }

        set_type(world.sigma(types));
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2
