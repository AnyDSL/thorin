#include "thorin/primop.h"

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/array.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * constructors
 */

PrimLit::PrimLit(World& world, PrimTypeKind kind, Box box, const std::string& name)
    : Literal((NodeKind) kind, world.type(kind), name)
    , box_(box)
{}

Cmp::Cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const std::string& name)
    : BinOp((NodeKind) kind, lhs->world().type_bool(lhs->type()->length()), cond, lhs, rhs, name)
{}

DefiniteArray::DefiniteArray(World& world, Type elem, ArrayRef<Def> args, const std::string& name)
    : Aggregate(Node_DefiniteArray, args, name)
{
    set_type(world.definite_array_type(elem, args.size()));
#ifndef NDEBUG
    for (size_t i = 0, e = size(); i != e; ++i)
        assert(args[i]->type() == type()->elem_type());
#endif
}

IndefiniteArray::IndefiniteArray(World& world, Type elem, Def dim, const std::string& name)
    : Aggregate(Node_IndefiniteArray, {dim}, name)
{
    set_type(world.indefinite_array_type(elem));
}

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

LEA::LEA(Def ptr, Def index, const std::string& name)
    : PrimOp(Node_LEA, Type(), {ptr, index}, name)
{
    auto& world = index->world();
    auto type = ptr_type();
    if (auto tuple = referenced_type().isa<TupleType>()) {
        set_type(world.ptr_type(tuple->elem(index), type->length(), type->device(), type->addr_space()));
    } else if (auto array = referenced_type().isa<ArrayType>()) {
        set_type(world.ptr_type(array->elem_type(), type->length(), type->device(), type->addr_space()));
    } else if (auto struct_app = referenced_type().isa<StructAppType>()) {
        set_type(world.ptr_type(struct_app->elem(index)));
    } else {
        THORIN_UNREACHABLE;
    }
}

Slot::Slot(Type type, Def frame, size_t index, const std::string& name)
    : PrimOp(Node_Slot, type->world().ptr_type(type), {frame}, name)
    , index_(index)
{
    assert(frame->type().isa<FrameType>());
}

Global::Global(Def init, bool is_mutable, const std::string& name)
    : PrimOp(Node_Global, init->type()->world().ptr_type(init->type()), {init}, name)
    , is_mutable_(is_mutable)
{ 
    assert(init->is_const());
}

Alloc::Alloc(Type type, Def mem, Def extra, const std::string& name)
    : MemOp(Node_Alloc, nullptr, {mem, extra}, name)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), w.ptr_type(type)}));
}

Load::Load(Def mem, Def ptr, const std::string& name)
    : Access(Node_Load, nullptr, {mem, ptr}, name)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), ptr->type().as<PtrType>()->referenced_type()}));
}

Enter::Enter(Def mem, const std::string& name)
    : MemOp(Node_Enter, nullptr, {mem}, name)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), w.frame_type()}));
}

Map::Map(int32_t device, AddressSpace addr_space, Def mem, Def ptr, Def mem_offset, Def mem_size, const std::string &name)
    : MapOp(Node_Map, Type(), {mem, ptr, mem_offset, mem_size}, name)
{
    World& w = mem->world();
    auto ptr_type = ptr->type().as<PtrType>();
    set_type(w.tuple_type({ w.mem_type(), w.ptr_type(ptr_type->referenced_type(), ptr_type->length(), device, addr_space)}));
}

//------------------------------------------------------------------------------

/*
 * hash
 */

size_t PrimOp::vhash() const {
    size_t seed = hash_combine(hash_combine(hash_begin((int) kind()), size()), type().unify()->gid());
    for (auto op : ops_)
        seed = hash_combine(seed, op.node()->gid());
    return seed;
}

size_t PrimLit::vhash() const { return hash_combine(Literal::vhash(), bcast<uint64_t, Box>(value())); }
size_t Slot::vhash() const { return hash_combine(PrimOp::vhash(), index()); }

//------------------------------------------------------------------------------

/*
 * equal
 */

bool PrimOp::equal(const PrimOp* other) const {
    bool result = this->kind() == other->kind() && this->size() == other->size() && this->type() == other->type();
    for (size_t i = 0, e = size(); result && i != e; ++i)
        result &= this->ops_[i].node() == other->ops_[i].node();
    return result;
}

bool PrimLit::equal(const PrimOp* other) const {
    return Literal::equal(other) ? this->value() == other->as<PrimLit>()->value() : false;
}

bool Slot::equal(const PrimOp* other) const {
    return PrimOp::equal(other) ? this->index() == other->as<Slot>()->index() : false;
}

//------------------------------------------------------------------------------

/*
 * getters
 */

PtrType LEA::ptr_type() const { return ptr()->type().as<PtrType>(); }
Type    LEA::referenced_type() const { return ptr_type()->referenced_type(); }
PtrType Slot::ptr_type() const { return type().as<PtrType>(); }
Type    Global::referenced_type() const { return type().as<PtrType>()->referenced_type(); }
Def     Alloc::extract_mem()   const { return world().extract(this, 0); }
Def     Alloc::extract_ptr()   const { return world().extract(this, 1); }
Def     Load ::extract_mem()   const { return world().extract(this, 0); }
Def     Load ::extract_val()   const { return world().extract(this, 1); }
Def     Enter::extract_mem()   const { return world().extract(this, 0); }
Def     Enter::extract_frame() const { return world().extract(this, 1); }
Def     Map  ::extract_mem()   const { return world().extract(this, 0); }
Def     Map  ::extract_ptr()   const { return world().extract(this, 1); }

//------------------------------------------------------------------------------

/*
 * op_name
 */

const char* Global::op_name() const { return is_mutable() ? "global_mutable" : "global_immutable"; }

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

/*
 * misc
 */

Def PrimOp::rebuild() const {
    if (!up_to_date_) {
        Array<Def> ops(size());
        for (size_t i = 0, e = size(); i != e; ++i)
            ops[i] = op(i)->rebuild();

        auto def = world().rebuild(this, ops);
        this->replace(def);
        return def;
    } else
        return this;
}

Type Extract::type(Def agg, Def index) {
    if (auto tuple = agg->type().isa<TupleType>())
        return tuple->elem(index);
    else if (auto array = agg->type().isa<ArrayType>())
        return array->elem_type();
    else if (auto vector = agg->type().isa<VectorType>())
        return vector->scalarize();
    else if (auto struct_app = agg->type().isa<StructAppType>())
        return struct_app->elem(index);

    assert(false && "TODO");
}

//------------------------------------------------------------------------------

}
