#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/world.h"

namespace thorin {

#define HENK_STRUCT_UNIFIER_NAME name
#define HENK_TABLE_NAME world
#define HENK_TABLE_TYPE World
#include "thorin/henk.cpp.h"

//------------------------------------------------------------------------------


const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return world().ptr_type(ptr->referenced_type());
    return world().type(as<PrimType>()->primtype_kind());
}

bool FnType::is_returning() const {
    bool ret = false;
    for (auto arg : args()) {
        switch (arg->order()) {
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

#if 0
static Type2Type type2type(const Type* type, Types args) {
    assert(type->num_type_params() == args.size());
    Type2Type map;
    for (size_t i = 0, e = args.size(); i != e; ++i)
        map[type->type_param(i)] = args[i];
    assert(map.size() == args.size());
    return map;
}
#endif

const IndefiniteArrayType* is_indefinite(const Type* type) {
    if (auto indefinite_array_type = type->isa<IndefiniteArrayType>())
        return indefinite_array_type;
    if (!type->empty())
        return is_indefinite(type->args().back());
    return nullptr;
}

bool use_lea(const Type* type) { return type->isa<StructType>() || type->isa<ArrayType>(); }

//------------------------------------------------------------------------------

/*
 * vrebuild
 */

//const Type* DefiniteArrayType  ::vrebuild(World& to, Types args) const { return to.definite_array_type(args[0], dim()); }
//const Type* FnType             ::vrebuild(World& to, Types args) const { return to.fn_type(args); }
//const Type* FrameType          ::vrebuild(World& to, Types     ) const { return to.frame_type(); }
//const Type* IndefiniteArrayType::vrebuild(World& to, Types args) const { return to.indefinite_array_type(args[0]); }
//const Type* MemType            ::vrebuild(World& to, Types     ) const { return to.mem_type(); }
//const Type* PrimType           ::vrebuild(World& to, Types     ) const { return to.type(primtype_kind(), length()); }
//const Type* TupleType          ::vrebuild(World& to, Types args) const { return to.tuple_type(args); }
//const Type* PtrType::vrebuild(World& to, Types args) const { return to.ptr_type(args.front(), length(), device(), addr_space()); }

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (uint64_t)device()), (uint64_t)addr_space();
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

const Type* stream_type_params(std::ostream& os, const Type* type) {
    return type; // TODO
#if 0
    if (type->num_type_params() == 0)
        return os;
    return stream_list(os, type->type_params(), [&](const TypeParam* type_param) {
        if (type_param)
            os << type_param;
        else
            os << "<null>";
    }, "[", "]");
#endif
}

static std::ostream& stream_type_args(std::ostream& os, const Type* type) {
   return stream_list(os, type->args(), [&](const Type* type) { os << type; }, "(", ")");
}

//static std::ostream& stream_type_elems(std::ostream& os, const Type* type) {
    //if (auto struct_app = type->isa<StructType>())
        //return stream_list(os, struct_app->args(), [&](const Type* type) { os << type; }, "{", "}");
    //return stream_type_args(os, type);
//}

std::ostream& MemType::stream(std::ostream& os) const { return os << "mem"; }
std::ostream& FrameType::stream(std::ostream& os) const { return os << "frame"; }

std::ostream& FnType::stream(std::ostream& os) const {
    os << "fn";
    stream_type_params(os, this);
    return stream_type_args(os, this);
}

std::ostream& TupleType::stream(std::ostream& os) const {
  stream_type_params(os, this);
  return stream_type_args(os, this);
}

std::ostream& StructType::stream(std::ostream& os) const { return os << name(); }
std::ostream& TypeParam::stream(std::ostream& os) const { return os << name_; }
std::ostream& IndefiniteArrayType::stream(std::ostream& os) const { return streamf(os, "[%]", elem_type()); }
std::ostream& DefiniteArrayType::stream(std::ostream& os) const { return streamf(os, "[% x %]", dim(), elem_type()); }

std::ostream& PtrType::stream(std::ostream& os) const {
    if (is_vector())
        os << '<' << length() << " x ";
    os << referenced_type() << '*';
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

    switch (primtype_kind()) {
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
 * specialize
 */

const Type* FrameType::vspecialize(Type2Type& map) const { return map[this] = this; }
const Type* MemType  ::vspecialize(Type2Type& map) const { return map[this] = this; }
const Type* PrimType ::vspecialize(Type2Type& map) const { return map[this] = this; }

const Type* DefiniteArrayType::vspecialize(Type2Type& map) const {
    return map[this] = world().definite_array_type(elem_type()->specialize(map), dim());
}

const Type* FnType::vspecialize(Type2Type& map) const {
    return map[this] = world().fn_type(specialize_args(map));
}

const Type* IndefiniteArrayType::vspecialize(Type2Type& map) const {
    return map[this] = world().indefinite_array_type(elem_type()->specialize(map));
}

const Type* PtrType::vspecialize(Type2Type& map) const {
    return map[this] = world().ptr_type(referenced_type()->specialize(map), length(), device(), addr_space());
}

//------------------------------------------------------------------------------

}
