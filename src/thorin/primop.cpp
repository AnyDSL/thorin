#include "thorin/primop.h"
#include "thorin/continuation.h"

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
    : Literal((NodeTag) tag, world.prim_type(tag), dbg)
    , box_(box)
{}

Cmp::Cmp(CmpTag tag, const Def* lhs, const Def* rhs, Debug dbg)
    : BinOp((NodeTag) tag, (vector_length(lhs->type()) != 1) ? (Type*)lhs->world().vec_type(lhs->world().type_bool(), vector_length(lhs->type())) : (Type*)lhs->world().type_bool(1), lhs, rhs, dbg)
{}

DefiniteArray::DefiniteArray(World& world, const Type* elem, Defs args, Debug dbg)
    : Aggregate(Node_DefiniteArray, args, dbg)
{
    set_type(world.definite_array_type(elem, args.size()));
#if THORIN_ENABLE_CHECKS
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        assert(args[i]->type() == type()->elem_type());
#endif
}

IndefiniteArray::IndefiniteArray(World& world, const Type* elem, const Def* dim, Debug dbg)
    : Aggregate(Node_IndefiniteArray, {dim}, dbg)
{
    set_type(world.indefinite_array_type(elem));
}

Tuple::Tuple(World& world, Defs args, Debug dbg)
    : Aggregate(Node_Tuple, args, dbg)
{
    Array<const Type*> elems(num_ops());
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        elems[i] = args[i]->type();

    set_type(world.tuple_type(elems));
}

Vector::Vector(World& world, Defs args, Debug dbg)
    : Aggregate(Node_Vector, args, dbg)
{
    auto inner_type = args.front()->type();
    //if (auto primtype = inner_type->isa<PrimType>()) {
    //    assert(primtype->length() == 1);
    //    set_type(world.prim_type(primtype->primtype_tag(), args.size()));
    //} else if (auto ptrtype = inner_type->isa<PtrType>()) {
    //    assert(ptrtype->length() == 1);
    //    set_type(world.ptr_type(ptrtype->pointee(), args.size()));
    //} else {
    set_type(world.vec_type(inner_type, args.size()));
    //}
}

LEA::LEA(const Def* ptr, const Def* index, Debug dbg)
    : PrimOp(Node_LEA, nullptr, {ptr, index}, dbg)
{
    auto& world = index->world();
    auto type = ptr_type();
    const Type* inner_type;
    const PtrType* ptrtype;
    auto index_vector = index->type()->isa<VectorType>();

    if (auto typevec = type->isa<VectorType>(); typevec && typevec->is_vector()) {
        if (index_vector && index_vector->is_vector())
            assert(typevec->length() == index_vector->length());
        ptrtype = typevec->scalarize()->as<PtrType>();
    } else {
        ptrtype = type->as<PtrType>();
    }

    inner_type = ptr_pointee();

    if (auto vec_type = inner_type->isa<VectorType>(); vec_type && vec_type->is_vector()) {
        inner_type = vec_type->scalarize();
    }

    if (auto tuple = inner_type->isa<TupleType>()) {
        inner_type = get(tuple->ops(), index);
    } else if (auto array = inner_type->isa<ArrayType>()) {
        inner_type = array->elem_type();
    } else if (auto struct_type = inner_type->isa<StructType>()) {
        inner_type = get(struct_type->ops(), index);
    } else if (auto prim_type = inner_type->isa<PrimType>()) {
        assert(prim_type->length() > 1);
        inner_type = world.prim_type(prim_type->primtype_tag());
    } else {
        THORIN_UNREACHABLE;
    }

    if (index_vector && index_vector->is_vector()) {
        const Type * result_type;
        auto vector_width = index_vector->length();

        if (auto vectype = inner_type->isa<VectorType>()) {
            assert(vectype->length() == 1);
        }
        result_type = world.vec_type(world.ptr_type(inner_type, 1, ptrtype->device(), ptrtype->addr_space()), vector_width);
        set_type(result_type);
    } else {
        if (auto typevec = type->isa<VectorType>(); typevec && typevec->is_vector()) {
            inner_type = world.ptr_type(inner_type, 1, ptrtype->device(), ptrtype->addr_space());
            set_type(world.vec_type(inner_type, typevec->length()));
        } else {
            set_type(world.ptr_type(inner_type, 1, ptrtype->device(), ptrtype->addr_space()));
        }
    }
}

const Type* LEA::ptr_pointee() const {
    if (auto ptr_vector = ptr_type()->isa<VectorType>(); ptr_vector && ptr_vector->is_vector()) {
        auto& world = op(1)->world();
        return world.vec_type(ptr_vector->scalarize()->as<PtrType>()->pointee(), ptr_vector->length());
    } else
        return ptr_type()->as<PtrType>()->pointee();
}

Known::Known(const Def* def, Debug dbg)
    : PrimOp(Node_Known, def->world().type_bool(), {def}, dbg)
{}

AlignOf::AlignOf(const Def* def, Debug dbg)
    : PrimOp(Node_AlignOf, def->world().type_qs64(), {def}, dbg)
{}

SizeOf::SizeOf(const Def* def, Debug dbg)
    : PrimOp(Node_SizeOf, def->world().type_qs64(), {def}, dbg)
{}

Slot::Slot(const Type* type, const Def* frame, Debug dbg)
    : PrimOp(Node_Slot,
            type->isa<VectorType>() && type->as<VectorType>()->is_vector() ?
              (Type*) type->table().vec_type(type->table().ptr_type(type->as<VectorType>()->scalarize()), type->as<VectorType>()->length()) :
              (Type*) type->table().ptr_type(type),
            {frame},
            dbg)
{
    assert(frame->type()->isa<FrameType>());
}

Global::Global(const Def* init, bool is_mutable, Debug dbg)
    : PrimOp(Node_Global, init->type()->table().ptr_type(init->type()), {init}, dbg)
    , is_mutable_(is_mutable)
{
    assert(!init->has_dep(Dep::Param));
}

Alloc::Alloc(const Type* type, const Def* mem, const Def* extra, Debug dbg)
    : MemOp(Node_Alloc, nullptr, {mem, extra}, dbg)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), w.ptr_type(type)}));
}

Load::Load(const Def* mem, const Def* ptr, Debug dbg)
    : Access(Node_Load, nullptr, {mem, ptr}, dbg)
{
    World& w = mem->world();
    const Type* return_type;
    if (auto vec_type = ptr->type()->as<VectorType>(); vec_type && vec_type->is_vector()) {
        auto inner_type = vec_type->scalarize()->as<PtrType>()->pointee();
        return_type = w.vec_type(inner_type, vec_type->length());
    } else {
        return_type = ptr->type()->as<PtrType>()->pointee();
    }
    set_type(w.tuple_type({w.mem_type(), return_type}));
}

Enter::Enter(const Def* mem, Debug dbg)
    : MemOp(Node_Enter, nullptr, {mem}, dbg)
{
    World& w = mem->world();
    set_type(w.tuple_type({w.mem_type(), w.frame_type()}));
}

Assembly::Assembly(const Type *type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, Debug dbg)
    : MemOp(Node_Assembly, type, inputs, dbg)
    , asm_template_(asm_template)
    , output_constraints_(output_constraints)
    , input_constraints_(input_constraints)
    , clobbers_(clobbers)
    , flags_(flags)
{}

Variant::Variant(const Type* type, const Def* value, size_t index, Debug dbg)
    : PrimOp(Node_Variant, type, {value}, dbg), index_(index)
{
    if (auto variant_type = type->isa<VariantType>())
        assert(variant_type->op(index) == value->type());
    else if (auto vec_type = type->isa<VectorExtendedType>()) {
        auto variant_type = vec_type->element()->as<VariantType>();
        auto value_vector_type = value->type()->isa<VectorType>();
        if (!value_vector_type) {//Instead, we might see a tuple of vectors here.
            auto value_tuple_type = value->type()->as<TupleType>();
            Array<const Type*> elements(value_tuple_type->num_ops());
            for (size_t i = 0; i < value_tuple_type->num_ops(); ++i) {
                elements[i] = value_tuple_type->op(i)->as<VectorType>()->scalarize();
            }
            auto value_type =  world().tuple_type(elements);
            assert(variant_type->op(index) == value_type);
        } else {
            auto value_type = value_vector_type->scalarize();
            assert(variant_type->op(index) == value_type);
        }
    }
}

//------------------------------------------------------------------------------

/*
 * hash
 */

hash_t PrimOp::vhash() const {
    hash_t seed = hash_combine(hash_begin(uint8_t(tag())), uint32_t(type()->gid()));
    for (auto op : ops_)
        seed = hash_combine(seed, uint32_t(op->gid()));
    return seed;
}

hash_t Variant::vhash() const { return hash_combine(PrimOp::vhash(), index()); }
hash_t VariantExtract::vhash() const { return hash_combine(PrimOp::vhash(), index()); }
hash_t PrimLit::vhash() const { return hash_combine(Literal::vhash(), bitcast<uint64_t, Box>(value())); }
hash_t Slot::vhash() const { return hash_combine((int) tag(), gid()); }

//------------------------------------------------------------------------------

/*
 * equal
 */

bool PrimOp::equal(const PrimOp* other) const {
    bool result = this->tag() == other->tag() && this->num_ops() == other->num_ops() && this->type() == other->type();
    for (size_t i = 0, e = num_ops(); result && i != e; ++i)
        result &= this->ops_[i] == other->ops_[i];
    return result;
}

bool Variant::equal(const PrimOp* other) const {
    return PrimOp::equal(other) && other->as<Variant>()->index() == index();
}

bool VariantExtract::equal(const PrimOp* other) const {
    return PrimOp::equal(other) && other->as<VariantExtract>()->index() == index();
}

bool PrimLit::equal(const PrimOp* other) const {
    return Literal::equal(other) ? this->value() == other->as<PrimLit>()->value() : false;
}

bool Slot::equal(const PrimOp* other) const { return this == other; }

//------------------------------------------------------------------------------

/*
 * rebuild
 */

// do not use any of PrimOp's type getters - during import we need to derive types from 't' in the new world 'w'

const Def* ArithOp       ::rebuild(World& w, const Type*  , Defs o) const { return w.arithop(arithop_tag(), o[0], o[1], debug()); }
const Def* Bitcast       ::rebuild(World& w, const Type* t, Defs o) const { return w.bitcast(t, o[0], debug()); }
const Def* Bottom        ::rebuild(World& w, const Type* t, Defs  ) const { return w.bottom(t, debug()); }
const Def* Top           ::rebuild(World& w, const Type* t, Defs  ) const { return w.top(t, debug()); }
const Def* Cast          ::rebuild(World& w, const Type* t, Defs o) const { return w.cast(t, o[0], debug()); }
const Def* Cmp           ::rebuild(World& w, const Type*  , Defs o) const { return w.cmp(cmp_tag(), o[0], o[1], debug()); }
const Def* MathOp        ::rebuild(World& w, const Type*  , Defs o) const { return w.mathop(mathop_tag(), o, debug()); }
const Def* Enter         ::rebuild(World& w, const Type*  , Defs o) const { return w.enter(o[0], debug()); }
const Def* Extract       ::rebuild(World& w, const Type*  , Defs o) const { return w.extract(o[0], o[1], debug()); }
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
const Def* Slot          ::rebuild(World& w, const Type* t, Defs o) const {
    const Type *ttype;
    if (auto ptr = t->isa<PtrType>()) {
        ttype = ptr->pointee();
        auto vec_width = ptr->length();
        if (vec_width > 1)
            ttype = w.vec_type(ttype, vec_width);
    } else if (auto vec = t->isa<VectorType>(); vec && vec->is_vector()) {
        auto inner = vec->scalarize()->as<PtrType>();
        auto vec_width = vec->length();
        ttype = w.vec_type(inner->pointee(), vec_width);
    }
    return w.slot(ttype, o[0], debug());
}
const Def* Store         ::rebuild(World& w, const Type*  , Defs o) const { return w.store(o[0], o[1], o[2], debug()); }
const Def* Tuple         ::rebuild(World& w, const Type*  , Defs o) const { return w.tuple(o, debug()); }
const Def* Variant       ::rebuild(World& w, const Type* t, Defs o) const { return w.variant(t, o[0], index(), debug()); }
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
    return w.struct_agg(t, o, debug());
}

const Def* IndefiniteArray::rebuild(World& w, const Type* t, Defs o) const {
    return w.indefinite_array(t->as<IndefiniteArrayType>()->elem_type(), o[0], debug());
}

//------------------------------------------------------------------------------

/*
 * op_name
 */

const char* PrimOp::op_name() const {
    switch (tag()) {
#define THORIN_NODE(op, abbr) case Node_##op: return #abbr;
#include "thorin/tables/nodetable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* ArithOp::op_name() const {
    switch (tag()) {
#define THORIN_ARITHOP(op) case ArithOp_##op: return #op;
#include "thorin/tables/arithoptable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* Cmp::op_name() const {
    switch (tag()) {
#define THORIN_CMP(op) case Cmp_##op: return #op;
#include "thorin/tables/cmptable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* MathOp::op_name() const {
    switch (tag()) {
#define THORIN_MATHOP(op) case MathOp_##op: return #op;
#include "thorin/tables/mathoptable.h"
        default: THORIN_UNREACHABLE;
    }
}

const char* Global::op_name() const { return is_mutable() ? "global_mutable" : "global_immutable"; }

//------------------------------------------------------------------------------

/*
 * misc
 */

std::string DefiniteArray::as_string() const {
    std::string res;
    for (auto op : ops()) {
        auto c = op->as<PrimLit>()->pu8_value();
        if (!c) break;
        res += c;
    }
    return res;
}

const Def* PrimOp::out(size_t i) const {
    assert(i == 0 || i < type()->as<TupleType>()->num_ops());
    return world().extract(this, i, debug());
}

const Type* Extract::extracted_type(const Type* agg_type, const Def* index) {
    if (auto tupleindex = index->isa<Tuple>()) {
        assert(tupleindex->op(0)->isa<Top>());
        auto vector = agg_type->as<VectorType>();
        auto inner_type = vector->scalarize();

        auto element_type = extracted_type(inner_type, tupleindex->op(1));

        return index->world().vec_type(element_type, 8);
    }
    if (auto tuple = agg_type->isa<TupleType>()) {
        return get(tuple->ops(), index);
    } else if (auto array = agg_type->isa<ArrayType>())
        return array->elem_type();
    else if (auto vector = agg_type->isa<VectorType>())
        return vector->scalarize();
    else if (auto struct_type = agg_type->isa<StructType>())
        return get(struct_type->ops(), index);

    THORIN_UNREACHABLE;
}

bool is_from_match(const PrimOp* primop) {
    bool from_match = true;
    for (auto& use : primop->uses()) {
        if (auto continuation = use.def()->isa<Continuation>()) {
            auto callee = continuation->callee()->isa<Continuation>();
            if (callee && callee->intrinsic() == Intrinsic::Match) continue;
        }
        from_match = false;
    }
    return from_match;
}

const Type* Closure::environment_type(World& world) {
    // We assume that ptrs are <= 64 bits, if they're not, god help you
    return world.type_qu64();
}

const PtrType* Closure::environment_ptr_type(World& world) {
    return world.ptr_type(world.type_pu8());
}

const Type* Slot::alloced_type() const {
    if (auto ptr = type()->isa<PtrType>())
        return ptr->pointee();
    else if (auto vec = type()->isa<VectorType>(); vec && vec->is_vector()) {
        auto element = vec->scalarize()->as<PtrType>();
        auto vector_width = vec->length();
        return world().vec_type(element->pointee(), vector_width);
    } else
        THORIN_UNREACHABLE;
}

//------------------------------------------------------------------------------

}
