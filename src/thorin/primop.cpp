#include "thorin/primop.h"

#include "thorin/literal.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/array.h"

namespace thorin {

//------------------------------------------------------------------------------

size_t PrimOp::hash() const {
    size_t seed = hash_combine(hash_combine(hash_begin((int) kind()), size()), type()->gid());
    for (auto op : ops_)
        seed = hash_combine(seed, op.node()->gid());
    return seed;
}

//------------------------------------------------------------------------------

VectorOp::VectorOp(size_t size, NodeKind kind, Type type, Def cond, const std::string& name)
    : PrimOp(size, kind, type, name)
{
    assert(cond->type()->is_bool());
    set_op(0, cond);
}

BinOp::BinOp(NodeKind kind, Type type, Def cond, Def lhs, Def rhs, const std::string& name)
    : VectorOp(3, kind, type, cond, name)
{
    assert(lhs->type() == rhs->type() && "types are not equal");
    set_op(1, lhs);
    set_op(2, rhs);
}

Cmp::Cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const std::string& name)
    : BinOp((NodeKind) kind, lhs->world().type_bool(lhs->type().as<PrimType>()->length()), cond, lhs, rhs, name)
{
}

Select::Select(Def cond, Def tval, Def fval, const std::string& name)
    : VectorOp(3, Node_Select, tval->type(), cond, name)
{
    set_op(1, tval);
    set_op(2, fval);
    assert(tval->type() == fval->type() && "types of both values must be equal");
}

//------------------------------------------------------------------------------

DefiniteArray::DefiniteArray(World& world, Type elem, ArrayRef<Def> args, const std::string& name)
    : Aggregate(Node_DefiniteArray, args, name)
{
    set_type(world.definite_array_type(elem, args.size()));
#ifndef NDEBUG
    for (size_t i = 0, e = size(); i != e; ++i)
        assert(args[i]->type() == type()->elem_type());
#endif
}

Type DefiniteArray::elem_type() const { return type()->elem_type(); }

Tuple::Tuple(World& world, ArrayRef<Def> args, const std::string& name)
    : Aggregate(Node_Tuple, args, name)
{
    Array<Type> elems(size());
    for (size_t i = 0, e = size(); i != e; ++i)
        elems[i] = args[i]->type();

    set_type(world.tuple_type(elems));
}

Vector::Vector(World& world, ArrayRef<Def> args, const std::string& name)
    : Aggregate(Node_Vector, args, name)
{
    if (auto primtype = args.front()->type().isa<PrimType>()) {
        assert(primtype->length() == 1);
        set_type(world.type(primtype->primtype_kind(), args.size()));
    } else {
        PtrType ptr = args.front()->type().as<PtrType>();
        assert(ptr->length() == 1);
        set_type(world.ptr_type(ptr->referenced_type(), args.size()));
    }
}

TupleType Tuple::tuple_type() const { return type().as<TupleType>(); }

Extract::Extract(Def agg, Def index, const std::string& name)
    : AggOp(2, Node_Extract, type(agg, index), agg, index, name)
{}

Type Extract::type(Def agg, Def index) {
    if (auto tuple = agg->type().isa<TupleType>())
        return tuple->elem_via_lit(index);
    else if (auto array = agg->type().isa<ArrayType>())
        return array->elem_type();
    else if (auto vector = agg->type().isa<VectorType>())
        return vector->scalarize();
    assert(false && "TODO");
}

Insert::Insert(Def agg, Def index, Def value, const std::string& name)
    : AggOp(3, Node_Insert, type(agg), agg, index, name)
{
    set_op(2, value);
}

Type Insert::type(Def agg) { return agg->type(); }

//------------------------------------------------------------------------------

LEA::LEA(Def def, Def index, const std::string& name)
    : PrimOp(2, Node_LEA, Type(), name)
{
    set_op(0, def);
    set_op(1, index);

    auto type = ptr_type();
    if (auto tuple = referenced_type().isa<TupleType>())
        set_type(index->world().ptr_type(tuple->elem_via_lit(index), type->length(), type->device(), type->addr_space()));
    else {
        auto array = referenced_type().as<ArrayType>();
        set_type(index->world().ptr_type(array->elem_type(), type->length(), type->device(), type->addr_space()));
    }
}

PtrType LEA::ptr_type() const { return ptr()->type().as<PtrType>(); }
Type LEA::referenced_type() const { return ptr_type()->referenced_type(); }

//------------------------------------------------------------------------------

EvalOp::EvalOp(NodeKind kind, Def def, const std::string& name)
    : PrimOp(1, kind, def->type(), name)
{
    //assert(type()->isa<Pi>());
    set_op(0, def);
}

//------------------------------------------------------------------------------

Slot::Slot(Type type, Def frame, size_t index, const std::string& name)
    : PrimOp(1, Node_Slot, type->world().ptr_type(type), name)
    , index_(index)
{
    set_op(0, frame);
}

PtrType Slot::ptr_type() const { return type().as<PtrType>(); }

//------------------------------------------------------------------------------

Global::Global(Def init, bool is_mutable, const std::string& name)
    : PrimOp(1, Node_Global, init->type()->world().ptr_type(init->type()), name)
    , is_mutable_(is_mutable)
{
    set_op(0, init);
    assert(init->is_const());
}

Type Global::referenced_type() const { return type().as<PtrType>()->referenced_type(); }
const char* Global::op_name() const { return is_mutable() ? "global_mutable" : "global_immutable"; }

//------------------------------------------------------------------------------

const char* PrimOp::op_name() const {
    switch (kind()) {
#define THORIN_AIR_NODE(op, abbr) case Node_##op: return #abbr;
#include "thorin/tables/nodetable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* ArithOp::op_name() const {
    switch (kind()) {
#define THORIN_ARITHOP(op) case ArithOp_##op: return #op;
#include "thorin/tables/arithoptable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* Cmp::op_name() const {
    switch (kind()) {
#define THORIN_CMP(op) case Cmp_##op: return #op;
#include "thorin/tables/cmptable.h"
        default: THORIN_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

} // namespace thorin
