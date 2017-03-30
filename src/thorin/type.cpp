#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

#define HENK_STRUCT_EXTRA_NAME name
#define HENK_STRUCT_EXTRA_TYPE const char*
#define HENK_TABLE_NAME world
#define HENK_TABLE_TYPE World
#include "thorin/henk.cpp.h"

//------------------------------------------------------------------------------


const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return world().ptr_type(ptr->pointee());
    return world().type(as<PrimType>()->primtype_tag());
}

bool FnType::is_returning() const {
    bool ret = false;
    for (auto op : ops()) {
        switch (op->order()) {
            case 0: continue;
            case 1:
                if (!ret) {
                    ret = true;
                    continue;
                } // else fall-through
            default:
                return false;
        }
    }
    return true;
}

bool use_lea(const Type* type) { return type->isa<StructType>() || type->isa<ArrayType>(); }

//------------------------------------------------------------------------------

/*
 * vrebuild
 */

const Type* DefiniteArrayType  ::vrebuild(World& to, Types ops) const { return to.definite_array_type(ops[0], dim()); }
const Type* FnType             ::vrebuild(World& to, Types ops) const { return to.fn_type(ops); }
const Type* FrameType          ::vrebuild(World& to, Types    ) const { return to.frame_type(); }
const Type* IndefiniteArrayType::vrebuild(World& to, Types ops) const { return to.indefinite_array_type(ops[0]); }
const Type* MemType            ::vrebuild(World& to, Types    ) const { return to.mem_type(); }
const Type* PrimType           ::vrebuild(World& to, Types    ) const { return to.type(primtype_tag(), length()); }

const Type* PtrType::vrebuild(World& to, Types ops) const {
    return to.ptr_type(ops.front(), length(), device(), addr_space());
}

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (uint64_t)device(), (uint64_t)addr_space());
}

//------------------------------------------------------------------------------

/*
 * equal
 */

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
std::ostream& FrameType          ::stream(std::ostream& os) const { return os << "frame"; }
std::ostream& IndefiniteArrayType::stream(std::ostream& os) const { return streamf(os, "[{}]", elem_type()); }
std::ostream& Lambda             ::stream(std::ostream& os) const { return streamf(os, "[{}].{}", name(), body()); }
std::ostream& MemType            ::stream(std::ostream& os) const { return os << "mem"; }
std::ostream& StructType         ::stream(std::ostream& os) const { return os << name(); }
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

/*
 * reduce
 */

const Type* FrameType::vreduce(int, const Type*, Type2Type&) const { return this; }
const Type* MemType  ::vreduce(int, const Type*, Type2Type&) const { return this; }
const Type* PrimType ::vreduce(int, const Type*, Type2Type&) const { return this; }

const Type* DefiniteArrayType::vreduce(int depth, const Type* type, Type2Type& map) const {
    return world().definite_array_type(elem_type()->reduce(depth, type, map), dim());
}

const Type* FnType::vreduce(int depth, const Type* type, Type2Type& map) const {
    return world().fn_type(reduce_ops(depth, type, map));
}

const Type* IndefiniteArrayType::vreduce(int depth, const Type* type, Type2Type& map) const {
    return world().indefinite_array_type(elem_type()->reduce(depth, type, map));
}

const Type* PtrType::vreduce(int depth, const Type* type, Type2Type& map) const {
    return world().ptr_type(pointee()->reduce(depth, type, map), length(), device(), addr_space());
}

//------------------------------------------------------------------------------

}
