#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

/*
 * misc
 */

const Type* merge_tuple_type(const Type* a, const Type* b) {
    auto x = a->isa<TupleType>();
    auto y = b->isa<TupleType>();
    auto& w = a->table();

    if ( x &&  y) return w.tuple_type(concat(x->ops(), y->ops()));
    if ( x && !y) return w.tuple_type(concat(x->ops(), b       ));
    if (!x &&  y) return w.tuple_type(concat(a,        y->ops()));

    assert(!x && !y);
    return w.tuple_type({a, b});
}

//------------------------------------------------------------------------------

/*
 * vrebuild
 */

const Type* StructType::vrebuild(TypeTable&, Types ops) const {
    assert_unused(this->ops() == ops);
    return this;
}

const Type* App                ::vrebuild(TypeTable& to, Types ops) const { return to.app(ops[0], ops[1]); }
const Type* TupleType          ::vrebuild(TypeTable& to, Types ops) const { return to.tuple_type(ops); }
const Type* VariantType        ::vrebuild(TypeTable& to, Types ops) const { return to.variant_type(ops); }
const Type* Lambda             ::vrebuild(TypeTable& to, Types ops) const { return to.lambda(ops[0], name()); }
const Type* Var                ::vrebuild(TypeTable& to, Types    ) const { return to.var(depth()); }
const Type* DefiniteArrayType  ::vrebuild(TypeTable& to, Types ops) const { return to.definite_array_type(ops[0], dim()); }
const Type* FnType             ::vrebuild(TypeTable& to, Types ops) const { return to.fn_type(ops[0]); }
const Type* ClosureType        ::vrebuild(TypeTable& to, Types ops) const { return to.closure_type(ops[0]); }
const Type* FrameType          ::vrebuild(TypeTable& to, Types    ) const { return to.frame_type(); }
const Type* IndefiniteArrayType::vrebuild(TypeTable& to, Types ops) const { return to.indefinite_array_type(ops[0]); }
const Type* MemType            ::vrebuild(TypeTable& to, Types    ) const { return to.mem_type(); }
const Type* PrimType           ::vrebuild(TypeTable& to, Types    ) const { return to.type(primtype_tag(), length()); }

const Type* PtrType::vrebuild(TypeTable& to, Types ops) const {
    return to.ptr_type(ops.front(), length(), device(), addr_space());
}

//------------------------------------------------------------------------------

/*
 * reduce
 */

const Type* Lambda::vreduce(int depth, const Type* type, Type2Type& map) const {
    return table().lambda(body()->reduce(depth+1, type, map), name());
}

const Type* Var::vreduce(int depth, const Type* type, Type2Type&) const {
    if (this->depth() == depth)
        return type;
    else if (this->depth() > depth)
        return table().var(this->depth()-1);  // this is a free variable - shift by one
    else
        return this;                          // this variable is not free - don't adjust
}

const Type* StructType::vreduce(int depth, const Type* type, Type2Type& map) const {
    auto struct_type = table().struct_type(name(), num_ops());
    map[this] = struct_type;
    for (size_t i = 0, e = num_ops(); i != e; ++i)
        struct_type->set(i, op(i)->reduce(depth, type, map));

    return struct_type;
}

//------------------------------------------------------------------------------

const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return table().ptr_type(ptr->pointee());
    return table().type(as<PrimType>()->primtype_tag());
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

uint64_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (uint64_t)device(), (uint64_t)addr_space());
}

uint64_t Var::vhash() const {
    return murmur3(uint64_t(tag()) << uint64_t(56) | uint8_t(depth()));
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool Var::equal(const Type* other) const {
    return other->isa<Var>() ? this->as<Var>()->depth() == other->as<Var>()->depth() : false;
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

std::ostream& App                ::stream(std::ostream& os) const { return streamf(os, "{}[{}]", callee(), arg()); }
std::ostream& Var                ::stream(std::ostream& os) const { return streamf(os, "<{}>", depth()); }
std::ostream& DefiniteArrayType  ::stream(std::ostream& os) const { return streamf(os, "[{} x {}]", dim(), elem_type()); }
std::ostream& FnType             ::stream(std::ostream& os) const { return stream_type_ops(os << "fn", this); }
std::ostream& ClosureType        ::stream(std::ostream& os) const { return stream_type_ops(os << "closure", this); }
std::ostream& FrameType          ::stream(std::ostream& os) const { return os << "frame"; }
std::ostream& IndefiniteArrayType::stream(std::ostream& os) const { return streamf(os, "[{}]", elem_type()); }
std::ostream& Lambda             ::stream(std::ostream& os) const { return streamf(os, "[{}].{}", name(), body()); }
std::ostream& MemType            ::stream(std::ostream& os) const { return os << "mem"; }
std::ostream& StructType         ::stream(std::ostream& os) const { return os << name(); }
std::ostream& VariantType        ::stream(std::ostream& os) const { return stream_type_ops(os << "variant", this); }
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
    : unit_ (unify(new TupleType(*this, Types())))
    , fn0_  (unify(new FnType   (*this, unit_)))
    , mem_  (unify(new MemType  (*this)))
    , frame_(unify(new FrameType(*this)))
#define THORIN_ALL_TYPE(T, M) \
    , T##_(unify(new PrimType(*this, PrimType_##T, 1)))
#include "thorin/tables/primtypetable.h"
{}

const StructType* TypeTable::struct_type(Symbol name, size_t size) {
    auto type = new StructType(*this, name, size);
    const auto& p = types_.insert(type);
    assert_unused(p.second && "hash/equal broken");
    return type;
}

const Type* TypeTable::app(const Type* callee, const Type* op) {
    auto app = unify(new App(*this, callee, op));

    if (auto cache = app->cache_)
        return cache;
    if (auto lambda = app->callee()->template isa<Lambda>()) {
        Type2Type map;
        return app->cache_ = lambda->body()->reduce(1, op, map);
    } else {
        return app->cache_ = app;
    }

    return app;
}

//------------------------------------------------------------------------------

}
