#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>

#include "thorin/lambda.h"
#include "thorin/world.h"

namespace thorin {

//------------------------------------------------------------------------------

void TypeNode::bind(TypeParam type_param) const {
    assert(!type_param->is_unified());
    type_params_.push_back(type_param);
    type_param->bound_at_ = this;
}

void TypeNode::dump() const { std::cout << Type(this) << std::endl; }
size_t TypeNode::length() const { return as<VectorTypeNode>()->length(); }
Type TypeNode::elem(const Def& def) const { return elem(def->primlit_value<size_t>()); }

const TypeNode* TypeNode::unify() const {
    static const char* names[] = {"α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
                                  "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"};
    static const size_t num_names = sizeof(names)/sizeof(names[0]);

    bool first = !is_unified();
    auto type = world().unify_base(this);

    if (first) {
        for (size_t i = 0, e = type->num_type_params(); i != e; ++i) {
            auto type_param = type->type_param(i);
            for (size_t j = 0; j <= i / num_names; ++j)
                type_param->name_ += names[i % num_names];
        }
    }

    return type;
}

VectorType VectorTypeNode::scalarize() const {
    if (auto ptr = isa<PtrTypeNode>())
        return world().ptr_type(ptr->referenced_type());
    return world().type(as<PrimTypeNode>()->primtype_kind());
}

bool FnTypeNode::is_returning() const {
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

Type StructAppTypeNode::elem(size_t i) const {
    if (auto type = elem_cache_[i])
        return type;

    assert(i < struct_abs_type()->num_args());
    auto type = struct_abs_type()->arg(i);
    auto map = type2type(struct_abs_type(), type_args());
    return elem_cache_[i] = type->specialize(map).unify();
}

ArrayRef<Type> StructAppTypeNode::elems() const {
    for (size_t i = 0; i < num_elems(); ++i)
        elem(i);
    return elem_cache_;
}

//------------------------------------------------------------------------------

/*
 * vrebuild
 */

Type DefiniteArrayTypeNode  ::vrebuild(World& to, ArrayRef<Type> args) const { return to.definite_array_type(args[0], dim()); }
Type FnTypeNode             ::vrebuild(World& to, ArrayRef<Type> args) const { return to.fn_type(args); }
Type FrameTypeNode          ::vrebuild(World& to, ArrayRef<Type>     ) const { return to.frame_type(); }
Type IndefiniteArrayTypeNode::vrebuild(World& to, ArrayRef<Type> args) const { return to.indefinite_array_type(args[0]); }
Type MemTypeNode            ::vrebuild(World& to, ArrayRef<Type>     ) const { return to.mem_type(); }
Type PrimTypeNode           ::vrebuild(World& to, ArrayRef<Type>     ) const { return to.type(primtype_kind(), length()); }
Type TupleTypeNode          ::vrebuild(World& to, ArrayRef<Type> args) const { return to.tuple_type(args); }
Type TypeParamNode          ::vrebuild(World& to, ArrayRef<Type>     ) const { return to.type_param(); }

Type PtrTypeNode::vrebuild(World& to, ArrayRef<Type> args) const {
    return to.ptr_type(args.front(), length(), device(), addr_space());
}

Type StructAbsTypeNode::vrebuild(World& to, ArrayRef<Type> args) const {
    // TODO how do we handle recursive types?
    auto ntype = to.struct_abs_type(args.size());
    for (size_t i = 0, e = args.size(); i != e; ++i)
        ntype->set(i, args[i]);
    return ntype;
}

Type StructAppTypeNode::vrebuild(World& to, ArrayRef<Type> args) const {
    return to.struct_app_type(args[0].as<StructAbsType>(), args.skip_front());
}

//------------------------------------------------------------------------------

/*
 * recursive properties
 */

bool TypeNode::is_closed() const {
    for (auto arg : args()) {
        if (!arg->is_closed())
            return false;
    }
    return true;
}

IndefiniteArrayType TypeNode::is_indefinite() const {
    if (!empty())
        return args().back()->is_indefinite();
    return IndefiniteArrayType();
}

IndefiniteArrayType IndefiniteArrayTypeNode::is_indefinite() const { return this; }

TypeParamSet TypeNode::free_type_params() const { TypeParamSet bound, free; free_type_params(bound, free); return free; }

void TypeNode::free_type_params(TypeParamSet& bound, TypeParamSet& free) const {
    for (auto type_param : type_params())
        bound.insert(*type_param);

    for (auto arg : args()) {
        if (auto type_param = arg->isa<TypeParamNode>()) {
            if (!bound.contains(type_param))
                free.insert(type_param);
        } else
            arg->free_type_params(bound, free);
    }
}

bool TypeNode::is_concrete() const {
    for (auto arg : args()) {
        if (!arg->is_concrete())
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------

/*
 * hash
 */

uint64_t TypeNode::hash() const {
    uint64_t seed = hash_combine(hash_combine(hash_begin((int) kind()), num_args()), num_type_params());
    for (auto arg : args_)
        seed = hash_combine(seed, arg->hash());
    return seed;
}

uint64_t PtrTypeNode::hash() const {
    return hash_combine(hash_combine(VectorTypeNode::hash(), (uint64_t)device()), (uint64_t)addr_space());
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool TypeNode::equal(const TypeNode* other) const {
    bool result = this->kind() == other->kind() && this->num_args() == other->num_args()
        && this->num_type_params() == other->num_type_params();

    if (result) {
        for (size_t i = 0, e = num_type_params(); result && i != e; ++i) {
            assert(this->type_param(i)->equiv_ == nullptr);
            this->type_param(i)->equiv_ = *other->type_param(i);
        }

        for (size_t i = 0, e = num_args(); result && i != e; ++i)
            result &= this->args_[i]->equal(*other->args_[i]);

        for (auto var : type_params())
            var->equiv_ = nullptr;
    }

    return result;
}

bool PtrTypeNode::equal(const TypeNode* other) const {
    if(!VectorTypeNode::equal(other))
        return false;
    auto ptr = other->as<PtrTypeNode>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

bool TypeParamNode::equal(const TypeNode* other) const {
    if (auto type_param = other->isa<TypeParamNode>())
        return this->equiv_ == type_param;
    return false;
}

//------------------------------------------------------------------------------

/*
 * stream
 */

std::ostream& stream_type_params(std::ostream& os, Type type) {
   if (type->num_type_params() != 0)
       return stream_list(os, type->type_params(), [&](TypeParam type_param) { os << type_param; }, "[", "]");
   return os;
}

static std::ostream& stream_type_args(std::ostream& os, Type type) {
   return stream_list(os, type->args(), [&](Type type) { os << type; }, "(", ")");
}

static std::ostream& stream_type_elems(std::ostream& os, Type type) {
    if (auto struct_app = type.isa<StructAppType>())
        return stream_list(os, struct_app->elems(), [&](Type type) { os << type; }, "{", "}");
    return stream_type_args(os, type);
}

std::ostream& MemTypeNode::stream(std::ostream& os) const { return os << "mem"; }
std::ostream& FrameTypeNode::stream(std::ostream& os) const { return os << "frame"; }

std::ostream& FnTypeNode::stream(std::ostream& os) const {
    os << "fn";
    stream_type_params(os, this);
    return stream_type_args(os, this);
}

std::ostream& TupleTypeNode::stream(std::ostream& os) const {
  stream_type_params(os, this);
  return stream_type_args(os, this);
}

std::ostream& StructAbsTypeNode::stream(std::ostream& os) const {
    os << name();
    return stream_type_params(os, this);
    // TODO emit args - but don't do this inline: structs may be recursive
    //return emit_type_args(struct_abs);
}

std::ostream& StructAppTypeNode::stream(std::ostream& os) const {
    os << this->struct_abs_type()->name();
    return stream_type_elems(os, this);
}

std::ostream& TypeParamNode::stream(std::ostream& os) const { return os << name_; }
std::ostream& IndefiniteArrayTypeNode::stream(std::ostream& os) const { return streamf(os, "[%]", elem_type()); }
std::ostream& DefiniteArrayTypeNode::stream(std::ostream& os) const { return streamf(os, "[% x %]", dim(), elem_type()); }

std::ostream& PtrTypeNode::stream(std::ostream& os) const {
    if (is_vector())
        os << '<' << length() << " x ";
    os << referenced_type() << '*';
    if (is_vector())
        os << '>';
    if (device() != -1)
        os << '[' << device() << ']';
    switch (addr_space()) {
        case AddressSpace::Global:   os << "[Global]";   break;
        case AddressSpace::Texture:  os << "[Tex]";      break;
        case AddressSpace::Shared:   os << "[Shared]";   break;
        case AddressSpace::Constant: os << "[Constant]"; break;
        default: /* ignore unknown address space */      break;
    }
    return os;
}

std::ostream& PrimTypeNode::stream(std::ostream& os) const {
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

std::ostream& TypeNode::stream(std::ostream& os) const {
    assert(empty());
    return os << "<NULL>";
}

//------------------------------------------------------------------------------

/*
 * specialize and instantiate
 */

Type2Type type2type(const TypeNode* type, ArrayRef<Type> args) {
    assert(type->num_type_params() == args.size());
    Type2Type map;
    for (size_t i = 0, e = args.size(); i != e; ++i)
        map[*type->type_param(i)] = *args[i];
    assert(map.size() == args.size());
    return map;
}

Type TypeNode::instantiate(ArrayRef<Type> types) const {
    assert(types.size() == num_type_params());
    Type2Type map;
    for (size_t i = 0, e = types.size(); i != e; ++i)
        map[*type_param(i)] = *types[i];
    return instantiate(map);
}

Type TypeNode::instantiate(Type2Type& map) const {
#ifndef NDEBUG
    for (auto type_param : type_params())
        assert(map.contains(*type_param));
#endif
    return vinstantiate(map);
}

Type TypeNode::specialize(Type2Type& map) const {
    if (auto result = find(map, this))
        return result;

    for (auto type_param : type_params()) {
        assert(!map.contains(*type_param));
        map[*type_param] = *world().type_param();
    }

    auto t = instantiate(map);
    for (auto type_param : type_params())
        t->bind(map[*type_param]->as<TypeParamNode>());

    return t;
}

Array<Type> TypeNode::specialize_args(Type2Type& map) const {
    Array<Type> result(num_args());
    for (size_t i = 0, e = num_args(); i != e; ++i)
        result[i] = arg(i)->specialize(map);
    return result;
}

Type FrameTypeNode::vinstantiate(Type2Type& map) const { return map[this] = this; }
Type MemTypeNode  ::vinstantiate(Type2Type& map) const { return map[this] = this; }
Type PrimTypeNode ::vinstantiate(Type2Type& map) const { return map[this] = this; }
Type TypeParamNode::vinstantiate(Type2Type& map) const { return map[this] = this; }

Type DefiniteArrayTypeNode::vinstantiate(Type2Type& map) const {
    return map[this] = *world().definite_array_type(elem_type()->specialize(map), dim());
}

Type FnTypeNode::vinstantiate(Type2Type& map) const {
    return map[this] = *world().fn_type(specialize_args(map));
}

Type IndefiniteArrayTypeNode::vinstantiate(Type2Type& map) const {
    return map[this] = *world().indefinite_array_type(elem_type()->specialize(map));
}

Type PtrTypeNode::vinstantiate(Type2Type& map) const {
    return map[this] = *world().ptr_type(referenced_type()->specialize(map), length(), device(), addr_space());
}

Type StructAbsTypeNode::instantiate(ArrayRef<Type> args) const {
    return world().struct_app_type(this, args);
}

Type StructAppTypeNode::vinstantiate(Type2Type& map) const {
    Array<Type> nargs(num_type_args());
    for (size_t i = 0, e = num_type_args(); i != e; ++i)
        nargs[i] = type_arg(i)->specialize(map);
    return map[this] = *world().struct_app_type(struct_abs_type(), nargs);
}

Type TupleTypeNode::vinstantiate(Type2Type& map) const {
    return map[this] = *world().tuple_type(specialize_args(map));
}

//------------------------------------------------------------------------------

}
