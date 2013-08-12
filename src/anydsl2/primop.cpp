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
    for (auto op : ops())
        if (op)
            is_const_ &= op->is_const();
}

VectorOp::VectorOp(size_t size, NodeKind kind, const Type* type, const Def* cond, const std::string& name)
    : PrimOp(size, kind, type, name)
{
    assert(cond->type()->is_u1());
    set_op(0, cond);
}

BinOp::BinOp(NodeKind kind, const Type* type, const Def* cond, const Def* lhs, const Def* rhs, const std::string& name)
    : VectorOp(3, kind, type, cond, name)
{
    assert(lhs->type() == rhs->type() && "types are not equal");
    set_op(1, lhs);
    set_op(2, rhs);
}

RelOp::RelOp(RelOpKind kind, const Def* cond, const Def* lhs, const Def* rhs, const std::string& name)
    : BinOp((NodeKind) kind, lhs->world().type_u1(lhs->type()->as<PrimType>()->length()), cond, lhs, rhs, name)
{}

Select::Select(const Def* cond, const Def* tval, const Def* fval, const std::string& name)
    : VectorOp(3, Node_Select, tval->type(), cond, name)
{
    set_op(1, tval);
    set_op(2, fval);
    assert(tval->type() == fval->type() && "types of both values must be equal");
}

//------------------------------------------------------------------------------

TupleExtract::TupleExtract(const Def* tuple, const Def* index, const std::string& name)
    : TupleOp(2, Node_TupleExtract, tuple->type()->as<Sigma>()->elem_via_lit(index), tuple, index, name)
{}

TupleInsert::TupleInsert(const Def* tuple, const Def* index, const Def* value, const std::string& name)
    : TupleOp(2, Node_TupleInsert, tuple->type()->as<Sigma>()->elem_via_lit(index), tuple, index, name)
{
    set_op(2, value);
}

//------------------------------------------------------------------------------

Tuple::Tuple(World& world, ArrayRef<const Def*> args, const std::string& name)
    : PrimOp(args.size(), Node_Tuple, /*type: set later*/ nullptr, name)
{
    Array<const Type*> elems(size());
    for (size_t i = 0, e = size(); i != e; ++i) {
        set_op(i, args[i]);
        elems[i] = args[i]->type();
    }

    set_type(world.sigma(elems));
}

//------------------------------------------------------------------------------

Vector::Vector(World& world, ArrayRef<const Def*> args, const std::string& name)
    : PrimOp(args.size(), Node_Vector, /*type: set later*/ nullptr, name)
{
    size_t i = 0;
    for (auto arg : args)
        set_op(i++, arg);

    if (const PrimType* primtype = args.front()->type()->isa<PrimType>()) {
        assert(primtype->length() == 1);
        set_type(world.type(primtype->primtype_kind(), args.size()));
    } else {
        const Ptr* ptr = args.front()->type()->as<Ptr>();
        assert(ptr->length() == 1);
        set_type(world.ptr(ptr->referenced_type(), args.size()));
    }
}

//------------------------------------------------------------------------------

const char* PrimOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_AIR_NODE(op, abbr) case Node_##op: return #abbr;
#include "anydsl2/tables/nodetable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const char* ArithOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_ARITHOP(op) case ArithOp_##op: return #op;
#include "anydsl2/tables/arithoptable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const char* RelOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_RELOP(op) case RelOp_##op: return #op;
#include "anydsl2/tables/reloptable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const char* ConvOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_CONVOP(op) case ConvOp_##op: return #op;
#include "anydsl2/tables/convoptable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2
