#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/transform/rewrite.h"
#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

Type::Type(World& w, NodeTag tag, Defs args, Debug dbg) : Type(w, tag, w.star(), args, dbg) {
    // The overridden version of set_op is ignored in the Def ctor, because according to the C++ spec, virtuals are disabled in ctors (!)
    // So this is not actually duplicate code - for the nominal types Type::set_op will do what we want.
    for (auto& def : args)
        order_ = std::max(order_, def->order());
}
Type::Type(World& w, NodeTag tag, size_t size, Debug dbg) : Type(w, tag, w.star(), size, dbg) {}

void Type::set_op(size_t i, const Def* def) {
    Def::set_op(i, def);
    order_ = std::max(order_, def->order());
}

Array<const Def*> types2defs(ArrayRef<const Type*> types) {
    Array<const Def*> defs(types.size());
    size_t i = 0;
    for (auto type : types)
        defs[i++] = type->as<Def>();
    return defs;
}

Array<const Type*> defs2types(ArrayRef<const Def*> defs) {
    Array<const Type*> types(defs.size());
    size_t i = 0;
    for (auto type : defs)
        types[i++] = type->as<Type>();
    return types;
}

//------------------------------------------------------------------------------

/*
 * rebuild
 */

const Type* NominalType::rebuild(World& w, const Type* t, Defs o) const {
    THORIN_UNREACHABLE;
}

const Type* BottomType         ::rebuild(World& w, const Type* t, Defs o) const { return w.bottom_type(); }
const Type* ClosureType        ::rebuild(World& w, const Type* t, Defs o) const { return w.closure_type(defs2types(o)); }
const Type* DefiniteArrayType  ::rebuild(World& w, const Type* t, Defs o) const { return w.definite_array_type(o[0]->as<Type>(), dim()); }
const Type* FnType             ::rebuild(World& w, const Type* t, Defs o) const { return w.fn_type(defs2types(o)); }
const Type* FrameType          ::rebuild(World& w, const Type* t, Defs o) const { return w.frame_type(); }
const Type* IndefiniteArrayType::rebuild(World& w, const Type* t, Defs o) const { return w.indefinite_array_type(o[0]->as<Type>()); }
const Type* MemType            ::rebuild(World& w, const Type* t, Defs o) const { return w.mem_type(); }
const Type* PrimType           ::rebuild(World& w, const Type* t, Defs o) const { return w.prim_type(primtype_tag(), length()); }
const Type* PtrType            ::rebuild(World& w, const Type* t, Defs o) const { return w.ptr_type(o[0]->as<Type>(), length(), device(), addr_space()); }
const Type* TupleType          ::rebuild(World& w, const Type* t, Defs o) const { return w.tuple_type(defs2types(o)); }

/*
 * stub
 */

StructType* StructType::stub(Rewriter& rewriter) const {
    auto type = rewriter.dst().struct_type(name(), num_ops());
    std::copy(op_names_.begin(), op_names_.end(), type->op_names().begin());
    return type;
}

VariantType* VariantType::stub(Rewriter& rewriter) const {
    auto type = rewriter.dst().variant_type(name(), num_ops());
    std::copy(op_names_.begin(), op_names_.end(), type->op_names().begin());
    return type;
}

//------------------------------------------------------------------------------

const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return world().ptr_type(ptr->pointee());
    return world().prim_type(as<PrimType>()->primtype_tag());
}

bool FnType::is_returning() const {
    bool ret = false;
    for (auto op : ops()) {
        switch (op->order()) {
            case 1:
                if (!ret) {
                    ret = true;
                    continue;
                }
                return false;
            default: continue;
        }
    }
    return ret;
}

bool VariantType::has_payload() const {
    return !std::all_of(types().begin(), types().end(), is_type_unit);
}

bool use_lea(const Type* type) { return type->isa<StructType>() || type->isa<ArrayType>(); }

//------------------------------------------------------------------------------

/*
 * hash
 */

hash_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (hash_t)device(), (hash_t)addr_space());
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool PtrType::equal(const Def* other) const {
    if (!VectorType::equal(other))
        return false;
    auto ptr = other->as<PtrType>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

TypeTable::TypeTable(World& world)
    : world_(world)
    , star_     (world.put<Star>((world)))
    , unit_     (world.put<TupleType>(world, Defs(), Debug()))
    , fn0_      (world.put<FnType    >(world, Defs(), Node_FnType, Debug()))
    , bottom_ty_(world.put<BottomType>(world, Debug()))
    , mem_      (world.put<MemType   >(world, Debug()))
    , frame_    (world.put<FrameType >(world, Debug()))
{
#define THORIN_ALL_TYPE(T, M) \
    primtypes_[PrimType_##T - Begin_PrimType] = world.make<PrimType>(world, PrimType_##T, 1, Debug());
#include "thorin/tables/primtypetable.h"
}

const Type* World::tuple_type(Types ops) {
    return ops.size() == 1 ? ops.front()->as<Type>() : make<TupleType>(*this, types2defs(ops), Debug());
}

StructType* World::struct_type(Symbol name, size_t size) {
    return put<StructType>(*this, name, size, Debug());
}

VariantType* World::variant_type(Symbol name, size_t size) {
    return put<VariantType>(*this, name, size, Debug());
}

const PrimType* World::prim_type(PrimTypeTag tag, size_t length) {
    size_t i = tag - Begin_PrimType;
    assert(i < (size_t) Num_PrimTypes);
    return length == 1 ? types_.primtypes_[i] : make<PrimType>(*this, tag, length, Debug());
}

const PtrType* World::ptr_type(const Type* pointee, size_t length, int32_t device, AddrSpace addr_space) {
    return make<PtrType>(*this, pointee, length, device, addr_space, Debug());
}

const FnType*              World::fn_type(Types args) { return make<FnType>(*this, types2defs(args), Node_FnType, Debug()); }
const ClosureType*         World::closure_type(Types args) { return make<ClosureType>(*this, types2defs(args), Debug()); }
const DefiniteArrayType*   World::definite_array_type(const Type* elem, u64 dim) { return make<DefiniteArrayType>(*this, elem, dim, Debug()); }
const IndefiniteArrayType* World::indefinite_array_type(const Type* elem) { return make<IndefiniteArrayType>(*this, elem, Debug()); }

//------------------------------------------------------------------------------

}
