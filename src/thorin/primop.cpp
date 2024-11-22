#include "thorin/primop.h"
#include "thorin/continuation.h"
#include "thorin/transform/rewrite.h"

#include "thorin/config.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/util/array.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * constructors
 */

PrimLit::PrimLit(World& world, PrimTypeTag tag, Box box, Debug dbg)
    : Literal(world, (NodeTag) tag, world.prim_type(tag), dbg)
    , box_(box)
{}

Cmp::Cmp(CmpTag tag, World& world, const Def* lhs, const Def* rhs, Debug dbg)
    : BinOp(world, (NodeTag) tag, world.type_bool(vector_length(lhs->type())), lhs, rhs, dbg)
{}

DefiniteArray::DefiniteArray(World& world, const Type* elem, Defs args, Debug dbg)
    : Aggregate(world, Node_DefiniteArray, args, dbg)
{
    set_type(world.definite_array_type(elem, args.size()));
#if THORIN_ENABLE_CHECKS
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        assert(args[i]->type() == type()->elem_type());
#endif
}

IndefiniteArray::IndefiniteArray(World& world, const Type* elem, const Def* dim, Debug dbg)
    : Aggregate(world, Node_IndefiniteArray, {dim}, dbg)
{
    set_type(world.indefinite_array_type(elem));
}

Tuple::Tuple(World& world, Defs args, Debug dbg)
    : Aggregate(world, Node_Tuple, args, dbg)
{
    Array<const Type*> elems(num_ops());
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        elems[i] = args[i]->type();

    set_type(world.tuple_type(elems));
}

Vector::Vector(World& world, Defs args, Debug dbg)
    : Aggregate(world, Node_Vector, args, dbg)
{
    if (auto primtype = args.front()->type()->isa<PrimType>()) {
        assert(primtype->length() == 1);
        set_type(world.prim_type(primtype->primtype_tag(), args.size()));
    } else {
        auto ptr = args.front()->type()->as<PtrType>();
        assert(ptr->length() == 1);
        set_type(world.ptr_type(ptr->pointee(), args.size()));
    }
}

LEA::LEA(World& world, const Def* ptr, const Def* index, Debug dbg)
    : Def(world, Node_LEA, nullptr, {ptr, index}, dbg)
{
    auto type = ptr_type();
    if (auto tuple = ptr_pointee()->isa<TupleType>()) {
        set_type(world.ptr_type(get(tuple->types(), index), type->length(), type->addr_space()));
    } else if (auto array = ptr_pointee()->isa<ArrayType>()) {
        set_type(world.ptr_type(array->elem_type(), type->length(), type->addr_space()));
    } else if (auto struct_type = ptr_pointee()->isa<StructType>()) {
        set_type(world.ptr_type(get(struct_type->types(), index), type->length(), type->addr_space()));
    } else if (auto prim_type = ptr_pointee()->isa<PrimType>()) {
        assert(prim_type->length() > 1);
        set_type(world.ptr_type(world.prim_type(prim_type->primtype_tag()), type->length(), type->addr_space()));
    } else {
        THORIN_UNREACHABLE;
    }
}

Known::Known(World& world, const Def* def, Debug dbg)
    : Def(world, Node_Known, world.type_bool(), {def}, dbg)
{}

AlignOf::AlignOf(World& world, const Def* def, Debug dbg)
    : Def(world, Node_AlignOf, world.type_qs64(), {def}, dbg)
{}

SizeOf::SizeOf(World& world, const Def* def, Debug dbg)
    : Def(world, Node_SizeOf, world.type_qs64(), {def}, dbg)
{}

Slot::Slot(World& world, const Type* type, const Def* frame, Debug dbg)
    : Def(world, Node_Slot, world.ptr_type(type), {frame}, dbg)
{
    assert(frame->type()->isa<FrameType>());
}

Global::Global(World& world, const Def* init, bool is_mutable, Debug dbg)
    : Def(world, Node_Global, world.ptr_type(init->type()), {init}, dbg)
    , is_mutable_(is_mutable)
{
    assert(!init->has_dep(Dep::Param));
}

Alloc::Alloc(World& world, const Type* type, const Def* mem, const Def* extra, Debug dbg)
    : MemOp(world, Node_Alloc, nullptr, {mem, extra}, dbg)
{
    set_type(world.tuple_type({world.mem_type(), world.ptr_type(type)}));
}

Load::Load(World& world, const Def* mem, const Def* ptr, Debug dbg)
    : Access(world, Node_Load, nullptr, {mem, ptr}, dbg)
{
    set_type(world.tuple_type({world.mem_type(), ptr->type()->as<PtrType>()->pointee()}));
}

Enter::Enter(World& world, const Def* mem, Debug dbg)
    : MemOp(world, Node_Enter, nullptr, {mem}, dbg)
{
    set_type(world.tuple_type({world.mem_type(), world.frame_type()}));
}

Assembly::Assembly(World& world, const Type *type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, Debug dbg)
    : MemOp(world, Node_Assembly, type, inputs, dbg)
    , asm_template_(asm_template)
    , output_constraints_(output_constraints)
    , input_constraints_(input_constraints)
    , clobbers_(clobbers)
    , flags_(flags)
{}

//------------------------------------------------------------------------------

/*
 * hash
 */

hash_t Def::vhash() const {
    if (isa_nom()) return murmur3(gid());

    hash_t seed = hash_combine(hash_begin(uint8_t(tag())), uint32_t(type()->gid()));
    for (auto op : ops_)
        seed = hash_combine(seed, uint32_t(op->gid()));
    return seed;
}

hash_t Variant::vhash() const { return hash_combine(Def::vhash(), index()); }
hash_t VariantExtract::vhash() const { return hash_combine(Def::vhash(), index()); }
hash_t PrimLit::vhash() const { return hash_combine(Literal::vhash(), bitcast<uint64_t, Box>(value())); }
hash_t Slot::vhash() const { return hash_combine((int) tag(), gid()); }

//------------------------------------------------------------------------------

/*
 * equal
 */

bool Def::equal(const Def* other) const {
    if (isa_nom()) return this == other;

    bool result = this->tag() == other->tag() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->ops_[i] == other->ops_[i];
    return result;
}

bool Variant::equal(const Def* other) const {
    return Def::equal(other) && other->as<Variant>()->index() == index();
}

bool VariantExtract::equal(const Def* other) const {
    return Def::equal(other) && other->as<VariantExtract>()->index() == index();
}

bool PrimLit::equal(const Def* other) const {
    return Literal::equal(other) ? this->value() == other->as<PrimLit>()->value() : false;
}

bool Slot::equal(const Def* other) const { return this == other; }

//------------------------------------------------------------------------------

/*
 * rebuild
 */

const Def* App           ::rebuild(World& w, const Type*  , Defs o) const { return w.app(o[App::Ops::Callee], o.skip_front(App::Ops::FirstArg), debug()); }
const Def* ArithOp       ::rebuild(World& w, const Type*  , Defs o) const { return w.arithop(arithop_tag(), o[0], o[1], debug()); }
const Def* Bitcast       ::rebuild(World& w, const Type* t, Defs o) const { return w.bitcast(t, o[0], debug()); }
const Def* Bottom        ::rebuild(World& w, const Type* t, Defs  ) const { return w.bottom(t, debug()); }
const Def* Top           ::rebuild(World& w, const Type* t, Defs  ) const { return w.top(t, debug()); }
const Def* Cast          ::rebuild(World& w, const Type* t, Defs o) const { return w.cast(t, o[0], debug()); }
const Def* Cmp           ::rebuild(World& w, const Type*  , Defs o) const { return w.cmp(cmp_tag(), o[0], o[1], debug()); }
const Def* MathOp        ::rebuild(World& w, const Type*  , Defs o) const { return w.mathop(mathop_tag(), o, debug()); }
const Def* Enter         ::rebuild(World& w, const Type*  , Defs o) const { return w.enter(o[0], debug()); }
const Def* Extract       ::rebuild(World& w, const Type*  , Defs o) const { return w.extract(o[0], o[1], debug()); }
const Def* Filter        ::rebuild(World& w, const Type*,   Defs o) const { return w.filter(o, debug()); }
const Def* Global        ::rebuild(World& w, const Type*  , Defs o) const { return w.global(o[0], is_mutable(), debug()); }
const Def* Hlt           ::rebuild(World& w, const Type*  , Defs o) const { return w.hlt(o[0], debug()); }
const Def* Known         ::rebuild(World& w, const Type*  , Defs o) const { return w.known(o[0], debug()); }
const Def* Run           ::rebuild(World& w, const Type*  , Defs o) const { return w.run(o[0], debug()); }
const Def* Insert        ::rebuild(World& w, const Type*  , Defs o) const { return w.insert(o[0], o[1], o[2], debug()); }
const Def* LEA           ::rebuild(World& w, const Type*  , Defs o) const { return w.lea(o[0], o[1], debug()); }
const Def* Load          ::rebuild(World& w, const Type*  , Defs o) const { return w.load(o[0], o[1], debug()); }
const Def* PrimLit       ::rebuild(World& w, const Type*  , Defs  ) const { return w.literal(primtype_tag(), value(), debug()); }
const Def* Select        ::rebuild(World& w, const Type*  , Defs o) const { return w.select(o[0], o[1], o[2], debug()); }
const Def* AlignOf       ::rebuild(World& w, const Type*  , Defs o) const { return w.align_of(o[0]->type(), debug()); }
const Def* SizeOf        ::rebuild(World& w, const Type*  , Defs o) const { return w.size_of(o[0]->type(), debug()); }
const Def* Slot          ::rebuild(World& w, const Type* t, Defs o) const { return w.slot(t->as<PtrType>()->pointee(), o[0], debug()); }
const Def* Store         ::rebuild(World& w, const Type*  , Defs o) const { return w.store(o[0], o[1], o[2], debug()); }
const Def* Tuple         ::rebuild(World& w, const Type*  , Defs o) const { return w.tuple(o, debug()); }
const Def* Variant       ::rebuild(World& w, const Type* t, Defs o) const { return w.variant(t->as<VariantType>(), o[0], index(), debug()); }
const Def* VariantIndex  ::rebuild(World& w, const Type*  , Defs o) const { return w.variant_index(o[0], debug()); }
const Def* VariantExtract::rebuild(World& w, const Type*  , Defs o) const { return w.variant_extract(o[0], index(), debug()); }
const Def* Closure       ::rebuild(World& w, const Type* t, Defs o) const { return w.closure(t->as<ClosureType>(), o[0], o[1], debug()); }
const Def* Vector        ::rebuild(World& w, const Type*  , Defs o) const { return w.vector(o, debug()); }

const Def* Alloc::rebuild(World& w, const Type* t, Defs o) const {
    return w.alloc(t->as<TupleType>()->op(1)->as<PtrType>()->pointee(), o[0], o[1], debug());
}

const Def* Assembly::rebuild(World& w, const Type* t, Defs o) const {
    return w.assembly(t, o, asm_template(), output_constraints(), input_constraints(), clobbers(), flags(), debug());
}

const Def* DefiniteArray::rebuild(World& w, const Type* t, Defs o) const {
    return w.definite_array(t->as<DefiniteArrayType>()->elem_type(), o, debug());
}

const Def* StructAgg::rebuild(World& w, const Type* t, Defs o) const {
    return w.struct_agg(t->as<StructType>(), o, debug());
}

const Def* IndefiniteArray::rebuild(World& w, const Type* t, Defs o) const {
    return w.indefinite_array(t->as<IndefiniteArrayType>()->elem_type(), o[0], debug());
}

//------------------------------------------------------------------------------

/*
 * op_name
 */

const char* Def::op_name() const {
    switch (tag()) {
#define THORIN_GLUE(pre, next)
#define THORIN_NODE(op, abbr) case Node_##op: return #abbr;
#define THORIN_PRIMTYPE(T) case Node_PrimType_##T: return #T;
#define THORIN_ARITHOP(op) case ArithOp_##op: return #op;
#define THORIN_CMP(op) case Cmp_##op: return #op;
#define THORIN_MATHOP(op) case MathOp_##op: return #op;
#include "thorin/tables/allnodes.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* Global::op_name() const { return is_mutable() ? "global_mutable" : "global_immutable"; }

//------------------------------------------------------------------------------

/*
 * misc
 */

bool Global::is_external() const { return world().is_external(this); }

std::string DefiniteArray::as_string() const {
    std::string res;
    for (auto op : ops()) {
        auto c = op->as<PrimLit>()->pu8_value();
        if (!c) break;
        res += c;
    }
    return res;
}

const Def* Def::out(size_t i) const {
    assert(i == 0 || i < type()->as<TupleType>()->num_ops());
    return world().extract(this, i, debug());
}

const Type* Extract::extracted_type(const Def* agg, const Def* index) {
    if (auto tuple = agg->type()->isa<TupleType>())
        return get(tuple->types(), index);
    else if (auto array = agg->type()->isa<ArrayType>())
        return array->elem_type();
    else if (auto vector = agg->type()->isa<VectorType>())
        return vector->scalarize();
    else if (auto struct_type = agg->type()->isa<StructType>())
        return get(struct_type->types(), index);

    THORIN_UNREACHABLE;
}

const Enter* Enter::is_out_mem(const Def* def) {
    if (auto extract = def->isa_structural<Extract>())
        if (is_primlit(extract->index(), 0))
            if (auto enter = extract->agg()->isa_structural<Enter>())
                return enter;
    return nullptr;
}

const Type* Closure::environment_type(World& world) {
    // We assume that ptrs are <= 64 bits, if they're not, god help you
    return world.type_qu64();
}

const PtrType* Closure::environment_ptr_type(World& world) {
    return world.ptr_type(world.type_pu8());
}

Continuation* Closure::fn() const {
    return op(0)->as_nom<Continuation>();
}

//------------------------------------------------------------------------------

}
