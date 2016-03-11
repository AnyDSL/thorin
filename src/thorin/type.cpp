#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/typetable.h"

namespace thorin {

//------------------------------------------------------------------------------

size_t Type::gid_counter_ = 1;

const Type* close_base(const Type*& type, ArrayRef<const TypeParam*> type_params) {
    assert(type->num_type_params() == type_params.size());

    for (size_t i = 0, e = type->num_type_params(); i != e; ++i) {
        assert(!type_params[i]->is_closed());
        type->type_params_[i] = type_params[i];
        type->type_params_[i]->binder_ = type;
        type->type_params_[i]->closed_ = true;
        type->type_params_[i]->index_ = i;
    }

    std::stack<const Type*> stack;
    TypeSet done;

    auto push = [&](const Type* type) {
        if (!type->is_closed() && !done.contains(type) && !type->isa<TypeParam>()) {
            done.insert(type);
            stack.push(type);
            return true;
        }
        return false;
    };

    push(type);

    // TODO this is potentially quadratic when closing n types
    while (!stack.empty()) {
        auto type = stack.top();

        bool todo = false;
        for (auto arg : type->args())
            todo |= push(arg);

        if (!todo) {
            stack.pop();
            type->closed_ = true;
            for (size_t i = 0, e = type->num_args(); i != e && type->closed_; ++i)
                type->closed_ &= type->arg(i)->is_closed();
        }
    }

    return type = type->typetable().unify_base(type);
}

const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<PtrType>())
        return typetable().ptr_type(ptr->referenced_type());
    return typetable().type(as<PrimType>()->primtype_kind());
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

static Type2Type type2type(const Type* type, Types args) {
    assert(type->num_type_params() == args.size());
    Type2Type map;
    for (size_t i = 0, e = args.size(); i != e; ++i)
        map[type->type_param(i)] = args[i];
    assert(map.size() == args.size());
    return map;
}

const Type* StructAppType::elem(size_t i) const {
    if (auto type = elem_cache_[i])
        return type;

    assert(i < struct_abs_type()->num_args());
    auto type = struct_abs_type()->arg(i);
    auto map = type2type(struct_abs_type(), type_args());
    return elem_cache_[i] = type->specialize(map);
}

Types StructAppType::elems() const {
    for (size_t i = 0; i < num_elems(); ++i)
        elem(i);
    return elem_cache_;
}

const IndefiniteArrayType* is_indefinite(const Type* type) {
    if (auto indefinite_array_type = type->isa<IndefiniteArrayType>())
        return indefinite_array_type;
    if (!type->empty())
        return is_indefinite(type->args().back());
    return nullptr;
}

bool use_lea(const Type* type) {
    return type->isa<StructAppType>() || type->isa<ArrayType>();
}

//------------------------------------------------------------------------------

/*
 * vrebuild
 */

const Type* Type::rebuild(TypeTable& to, Types args) const {
    assert(num_args() == args.size());
    if (args.empty() && &typetable() == &to)
        return this;
    return vrebuild(to, args);
}

const Type* DefiniteArrayType  ::vrebuild(TypeTable& to, Types args) const { return to.definite_array_type(args[0], dim()); }
const Type* FnType             ::vrebuild(TypeTable& to, Types args) const { return to.fn_type(args, num_type_params()); }
const Type* FrameType          ::vrebuild(TypeTable& to, Types     ) const { return to.frame_type(); }
const Type* IndefiniteArrayType::vrebuild(TypeTable& to, Types args) const { return to.indefinite_array_type(args[0]); }
const Type* MemType            ::vrebuild(TypeTable& to, Types     ) const { return to.mem_type(); }
const Type* PrimType           ::vrebuild(TypeTable& to, Types     ) const { return to.type(primtype_kind(), length()); }
const Type* TupleType          ::vrebuild(TypeTable& to, Types args) const { return to.tuple_type(args); }
const Type* TypeParam          ::vrebuild(TypeTable& to, Types     ) const { return to.type_param(name()); }

const Type* PtrType::vrebuild(TypeTable& to, Types args) const {
    return to.ptr_type(args.front(), length(), device(), addr_space());
}

const Type* StructAbsType::vrebuild(TypeTable& to, Types args) const {
    // TODO how do we handle recursive types?
    auto ntype = to.struct_abs_type(args.size(), num_type_params());
    for (size_t i = 0, e = args.size(); i != e; ++i)
        ntype->set(i, args[i]);
    return ntype;
}

const Type* StructAppType::vrebuild(TypeTable& to, Types args) const {
    return to.struct_app_type(args[0]->as<StructAbsType>(), args.skip_front());
}

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t Type::vhash() const {
    uint64_t seed = hash_combine(hash_begin((int) kind()), num_args(), num_type_params());
    for (auto arg : args_)
        seed = hash_combine(seed, arg->hash());
    return seed;
}

uint64_t PtrType::vhash() const {
    return hash_combine(VectorType::vhash(), (uint64_t)device()), (uint64_t)addr_space();
}

uint64_t TypeParam::vhash() const {
    return hash_combine(hash_begin(int(kind())), index(), int(binder()->kind()), binder()->num_type_params(), binder()->num_args());
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool Type::equal(const Type* other) const {
    bool result =  this->kind() == other->kind()     &&  this->is_monomorphic() == other->is_monomorphic()
            && this->num_args() == other->num_args() && this->num_type_params() == other->num_type_params();

    if (result) {
        for (size_t i = 0, e = num_type_params(); result && i != e; ++i) {
            assert(this->type_param(i)->equiv_ == nullptr);
            this->type_param(i)->equiv_ = other->type_param(i);
        }

        for (size_t i = 0, e = num_args(); result && i != e; ++i)
            result &= this->arg(i)->is_hashed()
                ? this->arg(i) == other->arg(i)
                : this->arg(i)->equal(other->arg(i));

        for (auto type_param : type_params())
            type_param->equiv_ = nullptr;
    }

    return result;
}

bool PtrType::equal(const Type* other) const {
    if (!VectorType::equal(other))
        return false;
    auto ptr = other->as<PtrType>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

bool TypeParam::equal(const Type* other) const {
    if (auto type_param = other->isa<TypeParam>())
        return this->equiv_ == type_param;
    return false;
}

//------------------------------------------------------------------------------

/*
 * stream
 */

std::ostream& stream_type_params(std::ostream& os, const Type* type) {
    if (type->num_type_params() == 0)
        return os;
    return stream_list(os, type->type_params(), [&](const TypeParam* type_param) {
        if (type_param)
            os << type_param;
        else
            os << "<null>";
    }, "[", "]");
}

static std::ostream& stream_type_args(std::ostream& os, const Type* type) {
   return stream_list(os, type->args(), [&](const Type* type) { os << type; }, "(", ")");
}

static std::ostream& stream_type_elems(std::ostream& os, const Type* type) {
    if (auto struct_app = type->isa<StructAppType>())
        return stream_list(os, struct_app->elems(), [&](const Type* type) { os << type; }, "{", "}");
    return stream_type_args(os, type);
}

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

std::ostream& StructAbsType::stream(std::ostream& os) const {
    os << name();
    return stream_type_params(os, this);
    // TODO emit args - but don't do this inline: structs may be recursive
    //return emit_type_args(struct_abs);
}

std::ostream& StructAppType::stream(std::ostream& os) const {
    os << this->struct_abs_type()->name();
    return stream_type_elems(os, this);
}

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
 * specialize and instantiate
 */

const Type* Type::instantiate(Types types) const {
    assert(types.size() == num_type_params());
    Type2Type map;
    for (size_t i = 0, e = types.size(); i != e; ++i)
        map[type_param(i)] = types[i];
    return instantiate(map);
}

const Type* Type::instantiate(Type2Type& map) const {
#ifndef NDEBUG
    for (auto type_param : type_params())
        assert(map.contains(type_param));
#endif
    return vinstantiate(map);
}

const Type* Type::specialize(Type2Type& map) const {
    if (auto result = find(map, this))
        return result;

    Array<const TypeParam*> ntype_params(num_type_params());
    for (size_t i = 0, e = num_type_params(); i != e; ++i) {
        assert(!map.contains(type_param(i)));
        auto ntype_param = typetable().type_param(type_param(i)->name());
        map[type_param(i)] = ntype_param;
        ntype_params[i] = ntype_param;
    }

    auto type = instantiate(map);
    return close(type, ntype_params);
}

Array<const Type*> Type::specialize_args(Type2Type& map) const {
    Array<const Type*> result(num_args());
    for (size_t i = 0, e = num_args(); i != e; ++i)
        result[i] = arg(i)->specialize(map);
    return result;
}

const Type* FrameType::vinstantiate(Type2Type& map) const { return map[this] = this; }
const Type* MemType  ::vinstantiate(Type2Type& map) const { return map[this] = this; }
const Type* PrimType ::vinstantiate(Type2Type& map) const { return map[this] = this; }
const Type* TypeParam::vinstantiate(Type2Type& map) const { return map[this] = this; }

const Type* DefiniteArrayType::vinstantiate(Type2Type& map) const {
    return map[this] = typetable().definite_array_type(elem_type()->specialize(map), dim());
}

const Type* FnType::vinstantiate(Type2Type& map) const {
    return map[this] = typetable().fn_type(specialize_args(map));
}

const Type* IndefiniteArrayType::vinstantiate(Type2Type& map) const {
    return map[this] = typetable().indefinite_array_type(elem_type()->specialize(map));
}

const Type* PtrType::vinstantiate(Type2Type& map) const {
    return map[this] = typetable().ptr_type(referenced_type()->specialize(map), length(), device(), addr_space());
}

const Type* StructAbsType::instantiate(Types args) const {
    return typetable().struct_app_type(this, args);
}

const Type* StructAppType::vinstantiate(Type2Type& map) const {
    Array<const Type*> nargs(num_type_args());
    for (size_t i = 0, e = num_type_args(); i != e; ++i)
        nargs[i] = type_arg(i)->specialize(map);
    return map[this] = typetable().struct_app_type(struct_abs_type(), nargs);
}

const Type* TupleType::vinstantiate(Type2Type& map) const {
    return map[this] = typetable().tuple_type(specialize_args(map));
}

//------------------------------------------------------------------------------

}
