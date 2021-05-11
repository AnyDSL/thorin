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

const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return table().ptr_type(ptr->pointee());
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

bool VariantType::has_payload() const {
    return !std::all_of(ops().begin(), ops().end(), is_type_unit);
}

bool use_lea(const Type* type) { return type->isa<StructType>() || type->isa<ArrayType>(); }

//------------------------------------------------------------------------------

/*
 * hash
 */

hash_t Type::vhash() const {
    if (is_nominal())
        return thorin::murmur3(hash_t(tag()) << hash_t(32-8) | hash_t(gid()));
    hash_t seed = thorin::hash_begin(uint8_t(tag()));
    for (auto op : ops_)
        seed = thorin::hash_combine(seed, uint32_t(op->gid()));
    return seed;
}

hash_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (hash_t)device(), (hash_t)addr_space());
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

Stream& Type::stream(Stream& s) const {
    if (false) {}
    else if (isa<  MemType>()) return s.fmt("mem");
    else if (isa<FrameType>()) return s.fmt("frame");
    else if (auto t = isa<DefiniteArrayType>()) {
        return s.fmt("[{} x {}]", t->dim(), t->elem_type());
    } else if (auto t = isa<FnType>()) {
        return s.fmt("fn[{, }]", t->ops());
    } else if (auto t = isa<ClosureType>()) {
        return s.fmt("closure [{, }]", t->ops());
    } else if (auto t = isa<IndefiniteArrayType>()) {
        return s.fmt("[{}]", t->elem_type());
    } else if (auto t = isa<StructType>()) {
        return s.fmt("struct {}", t->name());
    } else if (auto t = isa<VariantType>()) {
        return s.fmt("variant {}", t->name());
    } else if (auto t = isa<TupleType>()) {
        return s.fmt("[{, }]", t->ops());
    } else if (auto t = isa<PtrType>()) {
        if (t->is_vector()) s.fmt("<{} x", t->length());
        s.fmt("{}*", t->pointee());
        if (t->is_vector()) s.fmt(">");
        if (t->device() != -1) s.fmt("[{}]", t->device());

        switch (t->addr_space()) {
            case AddrSpace::Global:   s.fmt("[Global]");   break;
            case AddrSpace::Texture:  s.fmt("[Tex]");      break;
            case AddrSpace::Shared:   s.fmt("[Shared]");   break;
            case AddrSpace::Constant: s.fmt("[Constant]"); break;
            default: /* ignore unknown address space */    break;
        }
        return s;
    } else if (auto t = isa<PrimType>()) {
        if (t->is_vector()) s.fmt("<{} x", t->length());

        switch (t->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case Node_PrimType_##T: s.fmt(#T); break;
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE;
        }

        if (t->is_vector()) s.fmt(">");
        return s;
    }
    THORIN_UNREACHABLE;
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
    size_t i = tag - Begin_PrimType;
    assert(i < (size_t) Num_PrimTypes);
    return length == 1 ? primtypes_[i] : insert<PrimType>(*this, tag, length);
}

const PtrType* TypeTable::ptr_type(const Type* pointee, size_t length, int32_t device, AddrSpace addr_space) {
    return insert<PtrType>(*this, pointee, length, device, addr_space);
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
