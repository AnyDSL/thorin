#include "anydsl2/primop.h"

#include "anydsl2/literal.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/hash.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

Select::Select(const DefTuple3& args, const std::string& name)
    : PrimOp(3, args.get<0>(), args.get<1>(), name)
{
    set_op(0, args.get<2>());
    set_op(1, args.get<3>());
    set_op(2, args.get<4>());
    assert(cond()->type() == world().type_u1() && "condition must be of u1 type");
    assert(tval()->type() == fval()->type() && "types of both values must be equal");
}

//------------------------------------------------------------------------------


Tuple::Tuple(const DefTupleN& args, const std::string& name)
    : PrimOp(args.get<2>().size(), args.get<0>(), args.get<1>(), name)
{
    size_t x = 0;
    for_all (arg, args.get<2>())
        set_op(x++, arg);
}

//------------------------------------------------------------------------------

} // namespace anydsl2
