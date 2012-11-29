#include "anydsl2/primop.h"

#include "anydsl2/literal.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/hash.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

Select::Select(const DefTuple3& tuple, const std::string& name)
    : PrimOp(3, tuple.get<0>(), tuple.get<1>(), name)
{
    set_op(0, tuple.get<2>());
    set_op(1, tuple.get<3>());
    set_op(2, tuple.get<4>());
    assert(cond()->type() == world().type_u1() && "condition must be of u1 type");
    assert(tval()->type() == fval()->type() && "types of both values must be equal");
}

//------------------------------------------------------------------------------

Insert::Insert(const Def* tuple, const Def* index, const Def* value, const std::string& name)
    : TupleOp(3, Node_Insert, tuple->type(), tuple, index, name)
{
    set_op(2, value);
    assert(tuple->type()->as<Sigma>()->elem_via_lit(index) == value->type() && "type error");
}

//------------------------------------------------------------------------------

Tuple::Tuple(World& world, ArrayRef<const Def*> args, const std::string& name)
    : PrimOp(args.size(), Node_Tuple, (const Type*) 0, name)
{
    if (ops().empty())
        set_type(world.sigma0());
    else {
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
