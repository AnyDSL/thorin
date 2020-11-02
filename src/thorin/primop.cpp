#include "thorin/primop.h"
#include "thorin/lam.h"

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
    : Literal((NodeTag) tag, world.type(tag), dbg)
    , box_(box)
{}

Cmp::Cmp(CmpTag tag, const Def* lhs, const Def* rhs, Debug dbg)
    : BinOp((NodeTag) tag, lhs->world().type_bool(vector_length(lhs->type())), lhs, rhs, dbg)
{}

DefiniteArray::DefiniteArray(World& world, const Type* elem, Defs args, Debug dbg)
    : Aggregate(Node_DefiniteArray, world.definite_array_type(elem, args.size()), args, dbg)
{
#if THORIN_ENABLE_CHECKS
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        assert(args[i]->type() == type()->elem_type());
#endif
}

IndefiniteArray::IndefiniteArray(World& world, const Type* elem, const Def* dim, Debug dbg)
    : Aggregate(Node_IndefiniteArray, world.indefinite_array_type(elem), {dim}, dbg)
{}

static const Type* infer_tuple_type(World& world, Defs ops) {
    Array<const Type*> elems(ops.size());
    for (size_t i = 0, e = ops.size(); i != e; ++i)
        elems[i] = ops[i]->type();

    return world.tuple_type(elems);
}

Tuple::Tuple(World& world, Defs args, Debug dbg)
    : Aggregate(Node_Tuple, infer_tuple_type(world, args), args, dbg)
{}

static const Type* infer_vector_type(World& world, Defs args) {
    if (auto primtype = args.front()->type()->isa<PrimType>()) {
        assert(primtype->length() == 1);
        return world.type(primtype->primtype_tag(), args.size());
    }

    auto ptr = args.front()->type()->as<PtrType>();
    assert(ptr->length() == 1);
    return world.ptr_type(ptr->pointee(), args.size());
}

Vector::Vector(World& world, Defs args, Debug dbg)
    : Aggregate(Node_Vector, infer_vector_type(world, args), args, dbg)
{}

static const Type* infer_lea_type(World& world, const Def* ptr, const Def* index) {
    auto ptr_type = ptr->type()->as<PtrType>();
    auto ptr_pointee = ptr_type->pointee();

    if (auto tuple = ptr_pointee->isa<TupleType>()) {
        return world.ptr_type(get(tuple->ops(), index), ptr_type->length(), ptr_type->device(), ptr_type->addr_space());
    } else if (auto array = ptr_pointee->isa<ArrayType>()) {
        return world.ptr_type(array->elem_type(), ptr_type->length(), ptr_type->device(), ptr_type->addr_space());
    } else if (auto struct_type = ptr_pointee->isa<StructType>()) {
        return world.ptr_type(get(struct_type->ops(), index));
    } else if (auto prim_type = ptr_pointee->isa<PrimType>()) {
        assert(prim_type->length() > 1);
        return world.ptr_type(world.type(prim_type->primtype_tag()));
    } else {
        THORIN_UNREACHABLE;
    }
}

LEA::LEA(const Def* ptr, const Def* index, Debug dbg)
    : PrimOp(Node_LEA, infer_lea_type(ptr->world(), ptr, index), {ptr, index}, dbg)
{}

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
    : PrimOp(Node_Slot, type->table().ptr_type(type), {frame}, dbg)
{
    assert(frame->type()->isa<FrameType>());
}

Global::Global(const Def* init, bool is_mutable, Debug dbg)
    : PrimOp(Node_Global, init->type()->table().ptr_type(init->type()), {init}, dbg)
    , is_mutable_(is_mutable)
{
    assert(is_const(init));
}

Alloc::Alloc(const Type* type, const Def* mem, const Def* extra, Debug dbg)
    : MemOp(Node_Alloc, mem->world().tuple_type({mem->world().mem_type(), mem->world().ptr_type(type)}), {mem, extra}, dbg)
{}

Load::Load(const Def* mem, const Def* ptr, Debug dbg)
    : Access(Node_Load, mem->world().tuple_type({mem->world().mem_type(), ptr->type()->as<PtrType>()->pointee()}), {mem, ptr}, dbg)
{}

Enter::Enter(const Def* mem, Debug dbg)
    : MemOp(Node_Enter, mem->world().tuple_type({mem->world().mem_type(), mem->world().frame_type()}), {mem}, dbg)
{}

Assembly::Assembly(const Type *type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Flags flags, Debug dbg)
    : MemOp(Node_Assembly, type, inputs, dbg)
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

uint64_t Variant::vhash() const { return hash_combine(Def::vhash(), index()); }
uint64_t VariantExtract::vhash() const { return hash_combine(Def::vhash(), index()); }
uint64_t PrimLit::vhash() const { return hash_combine(Literal::vhash(), bcast<uint64_t, Box>(value())); }
uint64_t Slot::vhash() const { return hash_combine((int) tag(), gid()); }

//------------------------------------------------------------------------------

/*
 * equal
 */

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
 * vrebuild
 */

// do not use any of PrimOp's type getters - during import we need to derive types from 't' in the new world 'to'

const Def* ArithOp       ::vrebuild(World& to, const Type*  , Defs ops) const { return to.arithop(arithop_tag(), ops[0], ops[1], debug()); }
const Def* Bitcast       ::vrebuild(World& to, const Type* t, Defs ops) const { return to.bitcast(t, ops[0], debug()); }
const Def* Bottom        ::vrebuild(World& to, const Type* t, Defs    ) const { return to.bottom(t, debug()); }
const Def* Top           ::vrebuild(World& to, const Type* t, Defs    ) const { return to.top(t, debug()); }
const Def* Cast          ::vrebuild(World& to, const Type* t, Defs ops) const { return to.cast(t, ops[0], debug()); }
const Def* Cmp           ::vrebuild(World& to, const Type*  , Defs ops) const { return to.cmp(cmp_tag(), ops[0], ops[1], debug()); }
const Def* Enter         ::vrebuild(World& to, const Type*  , Defs ops) const { return to.enter(ops[0], debug()); }
const Def* Extract       ::vrebuild(World& to, const Type*  , Defs ops) const { return to.extract(ops[0], ops[1], debug()); }
const Def* Global        ::vrebuild(World& to, const Type*  , Defs ops) const { return to.global(ops[0], is_mutable(), debug()); }
const Def* Hlt           ::vrebuild(World& to, const Type*  , Defs ops) const { return to.hlt(ops[0], debug()); }
const Def* Known         ::vrebuild(World& to, const Type*  , Defs ops) const { return to.known(ops[0], debug()); }
const Def* Run           ::vrebuild(World& to, const Type*  , Defs ops) const { return to.run(ops[0], debug()); }
const Def* Insert        ::vrebuild(World& to, const Type*  , Defs ops) const { return to.insert(ops[0], ops[1], ops[2], debug()); }
const Def* LEA           ::vrebuild(World& to, const Type*  , Defs ops) const { return to.lea(ops[0], ops[1], debug()); }
const Def* Load          ::vrebuild(World& to, const Type*  , Defs ops) const { return to.load(ops[0], ops[1], debug()); }
const Def* PrimLit       ::vrebuild(World& to, const Type*  , Defs    ) const { return to.literal(primtype_tag(), value(), debug()); }
const Def* Select        ::vrebuild(World& to, const Type*  , Defs ops) const { return to.select(ops[0], ops[1], ops[2], debug()); }
const Def* AlignOf       ::vrebuild(World& to, const Type*  , Defs ops) const { return to.align_of(ops[0]->type(), debug()); }
const Def* SizeOf        ::vrebuild(World& to, const Type*  , Defs ops) const { return to.size_of(ops[0]->type(), debug()); }
const Def* Slot          ::vrebuild(World& to, const Type* t, Defs ops) const { return to.slot(t->as<PtrType>()->pointee(), ops[0], debug()); }
const Def* Store         ::vrebuild(World& to, const Type*  , Defs ops) const { return to.store(ops[0], ops[1], ops[2], debug()); }
const Def* Tuple         ::vrebuild(World& to, const Type*  , Defs ops) const { return to.tuple(ops, debug()); }
const Def* Variant       ::vrebuild(World& to, const Type* t, Defs ops) const { return to.variant(t->as<VariantType>(), ops[0], index(), debug()); }
const Def* VariantIndex  ::vrebuild(World& to, const Type*  , Defs ops) const { return to.variant_index(ops[0], debug()); }
const Def* VariantExtract::vrebuild(World& to, const Type*  , Defs ops) const { return to.variant_extract(ops[0], index(), debug()); }
const Def* Closure       ::vrebuild(World& to, const Type* t, Defs ops) const { return to.closure(t->as<Pi>(), ops[0], ops[1], debug()); }
const Def* Vector        ::vrebuild(World& to, const Type*  , Defs ops) const { return to.vector(ops, debug()); }

const Def* Alloc::vrebuild(World& to, const Type* t, Defs ops) const {
    return to.alloc(t->as<TupleType>()->op(1)->as<PtrType>()->pointee(), ops[0], ops[1], debug());
}

const Def* Assembly::vrebuild(World& to, const Type* t, Defs ops) const {
    return to.assembly(t, ops, asm_template(), output_constraints(), input_constraints(), clobbers(), flags(), debug());
}

const Def* DefiniteArray::vrebuild(World& to, const Type* t, Defs ops) const {
    return to.definite_array(t->as<DefiniteArrayType>()->elem_type(), ops, debug());
}

const Def* StructAgg::vrebuild(World& to, const Type* t, Defs ops) const {
    return to.struct_agg(t->as<StructType>(), ops, debug());
}

const Def* IndefiniteArray::vrebuild(World& to, const Type* t, Defs ops) const {
    return to.indefinite_array(t->as<IndefiniteArrayType>()->elem_type(), ops[0], debug());
}

//------------------------------------------------------------------------------

/*
 * op_name
 */

const char* Def::op_name() const {
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

const char* Global::op_name() const { return is_mutable() ? "global_mutable" : "global_immutable"; }

//------------------------------------------------------------------------------

/*
 * stream
 */

std::ostream& PrimOp::stream(std::ostream& os) const {
    if (is_const(this)) {
        if (num_ops() == 0)
            return streamf(os, "{} {}", op_name(), type());
        else
            return streamf(os, "({} {} {})", type(), op_name(), stream_list(ops(), [&](const Def* def) { os << def; }));
    } else
        return os << unique_name();
}

std::ostream& PrimLit::stream(std::ostream& os) const {
    os << type() << ' ';
    auto tag = primtype_tag();

    // print i8 as ints
    switch (tag) {
        case PrimType_qs8: return os << (int) qs8_value();
        case PrimType_ps8: return os << (int) ps8_value();
        case PrimType_qu8: return os << (unsigned) qu8_value();
        case PrimType_pu8: return os << (unsigned) pu8_value();
        default:
            switch (tag) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return os << value().get_##M();
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
    }
}

std::ostream& Global::stream(std::ostream& os) const { return os << unique_name(); }

std::ostream& Assembly::stream_assignment(std::ostream& os) const {
    streamf(os, "{} {} = asm \"{}\"", type(), unique_name(), asm_template());
    stream_list(os, output_constraints(), [&](const auto& output_constraint) { os << output_constraint; }, " : (", ")");
    stream_list(os,  input_constraints(), [&](const auto&  input_constraint) { os <<  input_constraint; }, " : (", ")");
    stream_list(os,           clobbers(), [&](const auto&           clobber) { os <<           clobber; }, " : (", ") ");
    return stream_list(os,         ops(), [&](const Def*                def) { os <<               def; },    "(", ")") << endl;
}

//------------------------------------------------------------------------------

/*
 * misc
 */

const Def* merge_tuple(const Def* a, const Def* b) {
    auto x = a->isa<Tuple>();
    auto y = b->isa<Tuple>();
    auto& w = a->world();

    if ( x &&  y) return w.tuple(concat(x->ops(), y->ops()));
    if ( x && !y) return w.tuple(concat(x->ops(), b       ));
    if (!x &&  y) return w.tuple(concat(a,        y->ops()));

    assert(!x && !y);
    return w.tuple({a, b});
}

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

const Type* Extract::extracted_type(const Def* agg, const Def* index) {
    if (auto tuple = agg->type()->isa<TupleType>())
        return get(tuple->ops(), index);
    else if (auto array = agg->type()->isa<ArrayType>())
        return array->elem_type();
    else if (auto vector = agg->type()->isa<VectorType>())
        return vector->scalarize();
    else if (auto struct_type = agg->type()->isa<StructType>())
        return get(struct_type->ops(), index);
    else {
        assert(index->as<PrimLit>()->value().get_u64() == 0);
        return agg->type();
    }
}

bool is_from_branch_or_match(const Def* def) {
    bool from_match = true;
    for (auto& use : def->uses()) {
        if (auto lam = use.def()->isa<Lam>()) {
            if (auto app = lam->body()->isa<App>()) {
                auto callee = app->callee()->isa<Lam>();
                if (callee && (callee->intrinsic() == Intrinsic::Branch || callee->intrinsic() == Intrinsic::Match)) continue;
            }
        }
        from_match = false;
    }
    return from_match;
}

const Type* Closure::environment_type(World& world) {
    // FIXME: This assumes that ptrs are <= 64 bits
    return world.type_qu64();
}

const PtrType* Closure::environment_ptr_type(World& world) {
    return world.ptr_type(world.type_pu8());
}

//------------------------------------------------------------------------------

}
