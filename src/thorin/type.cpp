#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

Type::Type(TypeTable& table, int tag, Types ops)
    : table_(&table)
    , tag_(tag)
    , ops_(ops.size())
{
    for (size_t i = 0, e = num_ops(); i != e; ++i) {
        if (auto op = ops[i])
            set(i, op);
    }
}

//------------------------------------------------------------------------------

/*
 * vrebuild
 */

const Type* NominalType::vrebuild(TypeTable&, Types) const {
    THORIN_UNREACHABLE;
    return this;
}

const Type* TupleType          ::vrebuild(TypeTable& to, Types ops) const { return to.tuple_type(ops); }
const Type* DefiniteArrayType  ::vrebuild(TypeTable& to, Types ops) const { return to.definite_array_type(ops[0], dim()); }
const Type* FnType             ::vrebuild(TypeTable& to, Types ops) const { return to.fn_type(ops); }
const Type* ClosureType        ::vrebuild(TypeTable& to, Types ops) const { return to.closure_type(ops); }
const Type* FrameType          ::vrebuild(TypeTable& to, Types    ) const { return to.frame_type(); }
const Type* IndefiniteArrayType::vrebuild(TypeTable& to, Types ops) const { return to.indefinite_array_type(ops[0]); }
const Type* MemType            ::vrebuild(TypeTable& to, Types    ) const { return to.mem_type(); }
const Type* PrimType           ::vrebuild(TypeTable& to, Types    ) const { return to.prim_type(primtype_tag(), length()); }

const Type* PtrType::vrebuild(TypeTable& to, Types ops) const {
    return to.ptr_type(ops.front(), length(), device(), addr_space());
}

const Type* VectorExtendedType::vrebuild(TypeTable& to, Types ops) const {
    return to.vec_type(ops.front(), length());
}

/*
 * stub
 */

const NominalType* StructType::stub(TypeTable& to) const {
    auto type = to.struct_type(name(), num_ops());
    std::copy(op_names_.begin(), op_names_.end(), type->op_names().begin());
    return type;
}

const NominalType* VariantType::stub(TypeTable& to) const {
    auto type = to.variant_type(name(), num_ops());
    std::copy(op_names_.begin(), op_names_.end(), type->op_names().begin());
    return type;
}

//------------------------------------------------------------------------------

const Type* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return table().ptr_type(ptr->pointee());
    if (auto vec = isa<VectorExtendedType>())
        return vec->element();
    return table().prim_type(as<PrimType>()->primtype_tag());
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

bool use_lea(const Type* type) { return type->isa<StructType>() || type->isa<ArrayType>(); }

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t Type::vhash() const {
    if (is_nominal())
        return thorin::murmur3(uint64_t(tag()) << uint64_t(56) | uint64_t(gid()));
    uint64_t seed = thorin::hash_begin(uint8_t(tag()));
    for (auto op : ops_)
        seed = thorin::hash_combine(seed, uint32_t(op->gid()));
    return seed;
}

uint64_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (uint64_t)device(), (uint64_t)addr_space());
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool Type::equal(const Type* other) const {
    if (is_nominal())
        return this == other;
    if (tag() == other->tag() && num_ops() == other->num_ops())
        return std::equal(ops().begin(), ops().end(), other->ops().begin());
    return false;
}

bool PtrType::equal(const Type* other) const {
    if (!VectorType::equal(other))
        return false;
    auto ptr = other->as<PtrType>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

//------------------------------------------------------------------------------

/*
 * stream
 */

static std::ostream& stream_type_ops(std::ostream& os, const Type* type) {
   return stream_list(os, type->ops(), [&](const Type* type) { os << type; }, "(", ")");
}

std::ostream& DefiniteArrayType  ::stream(std::ostream& os) const { return streamf(os, "[{} x {}]", dim(), elem_type()); }
std::ostream& FnType             ::stream(std::ostream& os) const { return stream_type_ops(os << "fn", this); }
std::ostream& ClosureType        ::stream(std::ostream& os) const { return stream_type_ops(os << "closure", this); }
std::ostream& FrameType          ::stream(std::ostream& os) const { return os << "frame"; }
std::ostream& IndefiniteArrayType::stream(std::ostream& os) const { return streamf(os, "[{}]", elem_type()); }
std::ostream& MemType            ::stream(std::ostream& os) const { return os << "mem"; }
std::ostream& StructType         ::stream(std::ostream& os) const { return os << "struct " << name(); }
std::ostream& VariantType        ::stream(std::ostream& os) const { return os << "variant " << name(); }
std::ostream& TupleType          ::stream(std::ostream& os) const { return stream_type_ops(os, this); }

std::ostream& PtrType::stream(std::ostream& os) const {
    if (is_vector())
        os << '<' << length() << " x ";
    os << pointee() << '*';
    if (is_vector())
        os << '>';
    if (device() != -1)
        os << '[' << device() << ']';
    switch (addr_space()) {
        case AddrSpace::Global:   os << "[Global]";   break;
        case AddrSpace::Texture:  os << "[Tex]";      break;
        case AddrSpace::Shared:   os << "[Shared]";   break;
        case AddrSpace::Constant: os << "[Constant]"; break;
        default: /* ignore unknown address space */      break;
    }
    return os;
}

std::ostream& VectorExtendedType::stream(std::ostream& os) const {
    if (is_vector())
        os << '<' << length() << " x ";
    os << element();
    if (is_vector())
        os << '>';
    return os;
}

std::ostream& PrimType::stream(std::ostream& os) const {
    if (is_vector())
        os << "<" << length() << " x ";

    switch (primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case Node_PrimType_##T: os << #T; break;
#include "thorin/tables/primtypetable.h"
          default: THORIN_UNREACHABLE;
    }

    if (is_vector())
        os << ">";

    return os;
}

//------------------------------------------------------------------------------

TypeTable::TypeTable()
    : unit_ (insert<TupleType>(*this, Types()))
    , fn0_  (insert<FnType   >(*this, Types()))
    , mem_  (insert<MemType  >(*this))
    , frame_(insert<FrameType>(*this))
{
#define THORIN_ALL_TYPE(T, M) \
    primtypes_[PrimType_##T - Begin_PrimType] = insert<PrimType>(*this, PrimType_##T, 1);
#include "thorin/tables/primtypetable.h"
}

const Type* TypeTable::tuple_type(Types ops) {
    return ops.size() == 1 ? ops.front() : insert<TupleType>(*this, ops);
}

const StructType* TypeTable::struct_type(Symbol name, size_t size) {
    auto type = new StructType(*this, name, size);
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

const VariantType* TypeTable::variant_type(Symbol name, size_t size) {
    auto type = new VariantType(*this, name, size);
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

const PrimType* TypeTable::prim_type(PrimTypeTag tag, size_t length) {
    assert(length == 1);
    size_t i = tag - Begin_PrimType;
    assert(i < (size_t) Num_PrimTypes);
    return length == 1 ? primtypes_[i] : insert<PrimType>(*this, tag, length);
}

const PtrType* TypeTable::ptr_type(const Type* pointee, size_t length, int32_t device, AddrSpace addr_space) {
    return insert<PtrType>(*this, pointee, length, device, addr_space);
}

const VectorExtendedType* TypeTable::vec_type(const Type* element, size_t length) {
    return insert<VectorExtendedType>(*this, element, length);
}


const FnType*              TypeTable::fn_type(Types args) { return insert<FnType>(*this, args); }
const ClosureType*         TypeTable::closure_type(Types args) { return insert<ClosureType>(*this, args); }
const DefiniteArrayType*   TypeTable::definite_array_type(const Type* elem, u64 dim) { return insert<DefiniteArrayType>(*this, elem, dim); }
const IndefiniteArrayType* TypeTable::indefinite_array_type(const Type* elem) { return insert<IndefiniteArrayType>(*this, elem); }

template <typename T, typename... Args>
const T* TypeTable::insert(Args&&... args) {
    T t(std::forward<Args&&>(args)...);
    auto it = types_.find(&t);
    if (it != types_.end())
        return (*it)->template as<T>();
    auto new_t = new T(std::move(t));
    new_t->gid_ = types_.size();
    types_.emplace(new_t);
    return new_t;
}

//------------------------------------------------------------------------------

}
