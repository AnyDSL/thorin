#include "anydsl2/primop.h"

#include "anydsl2/literal.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/hash.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

void PrimOp::update(size_t i, const Def* with) { 
    unset_op(i); 
    set_op(i, with); 

    is_const_ = true;
    for_all (op, ops())
        if (op)
            is_const_ &= op->is_const();
}

RelOp::RelOp(RelOpKind kind, const Def* lhs, const Def* rhs, const std::string& name)
    : BinOp((NodeKind) kind, lhs->world().type_u1(), lhs, rhs, name)
{}

Select::Select(const Def* cond, const Def* tval, const Def* fval, const std::string& name)
    : PrimOp(3, Node_Select, tval->type(), name)
{
    set_op(0, cond);
    set_op(1, tval);
    set_op(2, fval);
    assert(cond->type() == world().type_u1() && "condition must be of u1 type");
    assert(tval->type() == fval->type() && "types of both values must be equal");
}

//------------------------------------------------------------------------------

Extract::Extract(const Def* tuple, const Def* index, const std::string& name)
    : TupleOp(2, Node_Extract, tuple->type()->as<Sigma>()->elem_via_lit(index), tuple, index, name)
{}

Insert::Insert(const Def* tuple, const Def* index, const Def* value, const std::string& name)
    : TupleOp(2, Node_Insert, tuple->type()->as<Sigma>()->elem_via_lit(index), tuple, index, name)
{
    set_op(2, value);
}

Tuple::Tuple(World& world, ArrayRef<const Def*> args, const std::string& name)
    : PrimOp(args.size(), Node_Tuple, /*type: set later*/ 0, name)
{
    Array<const Type*> elems(size());
    size_t i = 0;
    for_all2 (arg, args, &elem, elems) {
        set_op(i++, arg);
        elem = arg->type();
    }

    set_type(world.sigma(elems));
}

//------------------------------------------------------------------------------

} // namespace anydsl2
