#include "thorin/primop.h"

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/array.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * constructors
 */

PrimLit::PrimLit(World& world, PrimTypeKind kind, Box box, const Location& loc, const std::string& name)
    : Literal((NodeKind) kind, world.type(kind), loc, name)
    , box_(box)
{}

Cmp::Cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const Location& loc, const std::string& name)
    : BinOp((NodeKind) kind, lhs->world().type_bool(lhs->type()->length()), cond, lhs, rhs, loc, name)
{}

DefiniteArray::DefiniteArray(World& world, Type elem, ArrayRef<Def> args, const Location& loc, const std::string& name)
    : Aggregate(Node_DefiniteArray, args, loc, name)
{
    set_type(world.definite_array_type(elem, args.size()));
#ifndef NDEBUG
    for (size_t i = 0, e = size(); i != e; ++i)
        assert(args[i]->type() == type()->elem_type());
#endif
}

IndefiniteArray::IndefiniteArray(World& world, Type elem, Def dim, const Location& loc, const std::string& name)
    : Aggregate(Node_IndefiniteArray, {dim}, loc, name)
{
    set_type(world.indefinite_array_type(elem));
}

Tuple::Tuple(World& world, ArrayRef<Def> args, const Location& loc, const std::string& name)
    : Aggregate(Node_Tuple, args, loc, name)
{
    Array<Type> elems(size());
    for (size_t i = 0, e = size(); i != e; ++i)
        elems[i] = args[i]->type();

    set_type(world.tuple_type(elems));
}

Vector::Vector(World& world, ArrayRef<Def> args, const Location& loc, const std::string& name)
    : Aggregate(Node_Vector, args, loc, name)
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

LEA::LEA(Def ptr, Def index, const Location& loc, const std::string& name)
    : PrimOp(Node_LEA, Type(), {ptr, index}, loc, name)
{
    auto& world = index->world();
    auto type = ptr_type();
    if (auto tuple = ptr_referenced_type().isa<TupleType>()) {
        set_type(world.ptr_type(tuple->elem(index), type->length(), type->device(), type->addr_space()));
    } else if (auto array = ptr_referenced_type().isa<ArrayType>()) {
        set_type(world.ptr_type(array->elem_type(), type->length(), type->device(), type->addr_space()));
    } else if (auto struct_app = ptr_referenced_type().isa<StructAppType>()) {
        set_type(world.ptr_type(struct_app->elem(index)));
    } else {
        THORIN_UNREACHABLE;
    }
}

Slot::Slot(Type type, Def frame, size_t index, const Location& loc, const std::string& name)
    : PrimOp(Node_Slot, type->world().ptr_type(type), {frame}, loc, name)
    , index_(index)
{
    assert(frame->type().isa<FrameType>());
}

Global::Global(Def init, bool is_mutable, const Location& loc, const std::string& name)
    : PrimOp(Node_Global, init->type()->world().ptr_type(init->type()), {init}, loc, name)
    , is_mutable_(is_mutable)
{
    assert(init->is_const());
}

Alloc::Alloc(Type type, Def mem, Def extra, const Location& loc, const std::string& name)
    : MemOp(Node_Alloc, nullptr, {mem, extra}, loc, name)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), w.ptr_type(type)}));
}

Load::Load(Def mem, Def ptr, const Location& loc, const std::string& name)
    : Access(Node_Load, nullptr, {mem, ptr}, loc, name)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), ptr->type().as<PtrType>()->referenced_type()}));
}

Enter::Enter(Def mem, const Location& loc, const std::string& name)
    : MemOp(Node_Enter, nullptr, {mem}, loc, name)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), w.frame_type()}));
}

Map::Map(int32_t device, AddressSpace addr_space, Def mem, Def ptr, Def mem_offset, Def mem_size, const Location& loc, const std::string &name)
    : Access(Node_Map, Type(), {mem, ptr, mem_offset, mem_size}, loc, name)
{
    World& w = mem->world();
    auto ptr_type = ptr->type().as<PtrType>();
    set_type(w.tuple_type({ w.mem_type(), w.ptr_type(ptr_type->referenced_type(), ptr_type->length(), device, addr_space)}));
}

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t PrimOp::vhash() const {
    uint64_t seed = hash_combine(hash_combine(hash_begin((int) kind()), size()), type().unify()->gid());
    for (auto op : ops_)
        seed = hash_combine(seed, op.node()->gid());
    return seed;
}

uint64_t PrimLit::vhash() const { return hash_combine(Literal::vhash(), bcast<uint64_t, Box>(value())); }
uint64_t Slot::vhash() const { return hash_combine(PrimOp::vhash(), index()); }

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
 * vrebuild
 */

// do not use any of PrimOp's type getters - during import we need to derive types from 't' in the new world 'to'

Def ArithOp::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.arithop(arithop_kind(), ops[0], ops[1], ops[2], this->loc(), name); }
Def Bitcast::vrebuild(World& to, ArrayRef<Def> ops, Type t) const { return to.bitcast(t, ops[0], ops[1], this->loc(), name); }
Def Bottom ::vrebuild(World& to, ArrayRef<Def>,     Type t) const { return to.bottom(t, this->loc()); }
Def Cast   ::vrebuild(World& to, ArrayRef<Def> ops, Type t) const { return to.cast(t, ops[0], ops[1], this->loc(), name); }
Def Cmp    ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.cmp(cmp_kind(), ops[0], ops[1], ops[2], this->loc(), name); }
Def Enter  ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.enter(ops[0], this->loc(), name); }
Def Extract::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.extract(ops[0], ops[1], this->loc(), name); }
Def Global ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.global(ops[0], this->loc(), is_mutable(), name); }
Def Hlt    ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.hlt(ops[0], this->loc(), name); }
Def Insert ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.insert(ops[0], ops[1], ops[2], this->loc(), name); }
Def LEA    ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.lea(ops[0], ops[1], this->loc(), name); }
Def Load   ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.load(ops[0], ops[1], this->loc(), name); }
Def PrimLit::vrebuild(World& to, ArrayRef<Def>,     Type  ) const { return to.literal(primtype_kind(), value(), this->loc()); }
Def Run    ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.run(ops[0], this->loc(), name); }
Def Select ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.select(ops[0], ops[1], ops[2], this->loc(), name); }
Def Store  ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.store(ops[0], ops[1], ops[2], this->loc(), name); }
Def Tuple  ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.tuple(ops, this->loc(), name); }
Def Vector ::vrebuild(World& to, ArrayRef<Def> ops, Type  ) const { return to.vector(ops, this->loc(), name); }

Def Alloc::vrebuild(World& to, ArrayRef<Def> ops, Type t) const {
    return to.alloc(t.as<TupleType>()->arg(1).as<PtrType>()->referenced_type(), ops[0], ops[1], this->loc(), name);
}

Def Slot::vrebuild(World& to, ArrayRef<Def> ops, Type t) const {
    return to.slot(t.as<PtrType>()->referenced_type(), ops[0], index(), this->loc(), name);
}

Def Map::vrebuild(World& to, ArrayRef<Def> ops, Type) const {
    return to.map(device(), addr_space(), ops[0], ops[1], ops[2], ops[3], this->loc(), name);
}

Def DefiniteArray::vrebuild(World& to, ArrayRef<Def> ops, Type t) const {
    return to.definite_array(t.as<DefiniteArrayType>()->elem_type(), ops, this->loc(), name);
}

Def StructAgg::vrebuild(World& to, ArrayRef<Def> ops, Type t) const {
    return to.struct_agg(t.as<StructAppType>(), ops, this->loc(), name);
}

Def IndefiniteArray::vrebuild(World& to, ArrayRef<Def> ops, Type t) const {
    return to.indefinite_array(t.as<IndefiniteArrayType>()->elem_type(), ops[0], this->loc(), name);
}

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
 * stream
 */

std::ostream& PrimOp::stream(std::ostream& os) const {
    // TODO is_const
    return os << this->unique_name();
}

std::ostream& PrimLit::stream(std::ostream& os) const {
    os << this->type() << ' ';
    auto kind = this->primtype_kind();

    // print i8 as ints
    if (kind == PrimType_qs8)
        os << (int) this->qs8_value();
    else if (kind == PrimType_ps8)
        os << (int) this->ps8_value();
    else if (kind == PrimType_qu8)
        os << (unsigned) this->qu8_value();
    else if (kind == PrimType_pu8)
        os << (unsigned) this->pu8_value();
    else {
        switch (kind) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: os << this->T##_value(); break;
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE;
        }
    }

    return os;
}

std::ostream& Global::stream(std::ostream& os) const {
    return PrimOp::stream(os);
}

//------------------------------------------------------------------------------

/*
 * misc
 */

Def PrimOp::out(size_t i) const {
    assert(i < type().as<TupleType>()->num_args());
    return world().extract(this, i, this->loc());
}

Def PrimOp::rebuild() const {
    if (is_outdated()) {
        Array<Def> ops(size());
        for (size_t i = 0, e = size(); i != e; ++i)
            ops[i] = op(i)->rebuild();

        auto def = rebuild(ops);
        this->replace(def);
        return def;
    } else
        return this;
}

Type Extract::extracted_type(Def agg, Def index) {
    if (auto tuple = agg->type().isa<TupleType>())
        return tuple->elem(index);
    else if (auto array = agg->type().isa<ArrayType>())
        return array->elem_type();
    else if (auto vector = agg->type().isa<VectorType>())
        return vector->scalarize();
    else if (auto struct_app = agg->type().isa<StructAppType>())
        return struct_app->elem(index);

    THORIN_UNREACHABLE;
}

//------------------------------------------------------------------------------

}
