#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"

namespace thorin {

class Def;
class IndefiniteArrayType;
class Type;
class TypeParam;
class World;

//------------------------------------------------------------------------------

template<class T>
struct GIDHash {
    uint64_t operator()(T n) const { return n->gid(); }
};

template<class T>
struct GIDEq {
    bool operator()(T n1, T n2) const { return n1->gid() == n2->gid(); }
};

template<class To>
using TypeMap    = HashMap<const Type*, To, GIDHash<const Type*>, GIDEq<const Type*>>;
using TypeSet    = HashSet<const Type*, GIDHash<const Type*>, GIDEq<const Type*>>;
using Type2Type  = TypeMap<const Type*>;
using TypeParamSet = HashSet<const TypeParam*, GIDHash<const TypeParam*>, GIDEq<const TypeParam*>>;

//Type2Type type2type(const Type*, Types);
//template<class T>
//Type2Type type2type(const Type* type, Types args) { return type2type(*type, args); }

typedef ArrayRef<const Type*> Types;

//------------------------------------------------------------------------------

/// Base class for all \p Type%s.
class Type : public MagicCast<Type> {
protected:
    Type(const Type&) = delete;
    Type& operator=(const Type&) = delete;

    Type(World& world, NodeKind kind, Types args)
        : representative_(nullptr)
        , world_(world)
        , kind_(kind)
        , args_(args.size())
        , gid_(gid_counter_++)
    {
        for (size_t i = 0, e = num_args(); i != e; ++i) {
            if (auto arg = args[i])
                set(i, arg);
        }
    }

    void set(size_t i, const Type* type) {
        args_[i] = type;
        order_ = std::max(order_, type->order());
    }

public:
    NodeKind kind() const { return kind_; }
    bool is_corenode() const { return ::thorin::is_corenode(kind()); }
    Types args() const { return args_; }
    ArrayRef<const TypeParam*> type_params() const { return type_params_; }
    size_t num_type_params() const { return type_params().size(); }
    const Type* arg(size_t i) const { assert(i < args().size()); return args()[i]; }
    const TypeParam* type_param(size_t i) const { assert(i < type_params().size()); return type_params()[i]; }
    void bind(const TypeParam*) const;
    size_t num_args() const { return args_.size(); }
    bool is_polymorphic() const { return num_type_params() > 0; }
    bool empty() const { return args_.empty(); }
    void dump() const;
    World& world() const { return world_; }
    const Type* representative() const { return representative_; }
    bool is_unified() const { return representative_ != nullptr; }
    const Type* unify() const;
    void free_type_params(TypeParamSet& bound, TypeParamSet& free) const;
    TypeParamSet free_type_params() const;
    size_t gid() const { return gid_; }
    int order() const { return order_; }
    /// Returns the vector length. Raises an assertion if this type is not a @p VectorType.
    size_t length() const;
    virtual const Type* instantiate(Types) const;
    const Type* instantiate(Type2Type&) const;
    const Type* specialize(Type2Type&) const;
    const Type* elem(const Def*) const;
    const Type* rebuild(World& to, Types args) const {
        assert(num_args() == args.size());
        if (args.empty() && &world() == &to)
            return this;
        return vrebuild(to, args);
    }
    const Type* rebuild(Types args) const { return rebuild(world(), args); }
    virtual const Type* elem(size_t i) const { return arg(i); }

    bool is_primtype() const { return thorin::is_primtype(kind()); }
    bool is_type_ps() const { return thorin::is_type_ps(kind()); }
    bool is_type_pu() const { return thorin::is_type_pu(kind()); }
    bool is_type_qs() const { return thorin::is_type_qs(kind()); }
    bool is_type_qu() const { return thorin::is_type_qu(kind()); }
    bool is_type_pf() const { return thorin::is_type_pf(kind()); }
    bool is_type_qf() const { return thorin::is_type_qf(kind()); }
    bool is_type_p() const { return thorin::is_type_p(kind()); }
    bool is_type_q() const { return thorin::is_type_q(kind()); }
    bool is_type_s() const { return thorin::is_type_s(kind()); }
    bool is_type_u() const { return thorin::is_type_u(kind()); }
    bool is_type_i() const { return thorin::is_type_i(kind()); }
    bool is_type_f() const { return thorin::is_type_f(kind()); }
    bool is_bool() const { return kind() == Node_PrimType_bool; }

    virtual uint64_t hash() const;
    virtual bool equal(const Type*) const;
    virtual bool is_closed() const;
    virtual bool is_concrete() const;
    virtual const IndefiniteArrayType* is_indefinite() const;
    virtual bool use_lea() const { return false; }
    virtual std::ostream& stream(std::ostream&) const;

    static size_t gid_counter() { return gid_counter_; }

protected:
    Array<const Type*> specialize_args(Type2Type&) const;

    int order_ = 0;

private:
    virtual const Type* vrebuild(World& to, Types args) const = 0;
    virtual const Type* vinstantiate(Type2Type&) const = 0;

    mutable const Type* representative_;
    World& world_;
    NodeKind kind_;
    mutable std::vector<const TypeParam*> type_params_;
    std::vector<const Type*> args_;
    mutable size_t gid_;
    static size_t gid_counter_;

    friend class World;
};

/// The type of the memory monad.
class MemType : public Type {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    MemType(World& world)
        : Type(world, Node_MemType, {})
    {}

    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class World;
};

/// The type of a stack frame.
class FrameType : public Type {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    FrameType(World& world)
        : Type(world, Node_FrameType, {})
    {}

    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class World;
};

/// Base class for all SIMD types.
class VectorType : public Type {
protected:
    VectorType(World& world, NodeKind kind, Types args, size_t length)
        : Type(world, kind, args)
        , length_(length)
    {}

    virtual uint64_t hash() const override { return hash_combine(Type::hash(), length()); }
    virtual bool equal(const Type* other) const override {
        return Type::equal(other) && this->length() == other->as<VectorType>()->length();
    }

public:
    /// The number of vector arguments - the vector length.
    size_t length() const { return length_; }
    bool is_vector() const { return length_ != 1; }
    /// Rebuilds the type with vector length 1.
    const VectorType* scalarize() const;

private:
    size_t length_;
};

/// Primitive type.
class PrimType : public VectorType {
private:
    PrimType(World& world, PrimTypeKind kind, size_t length)
        : VectorType(world, (NodeKind) kind, {}, length)
    {}

public:
    PrimTypeKind primtype_kind() const { return (PrimTypeKind) kind(); }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class World;
};

enum class AddressSpace : uint32_t {
    Generic  = 0,
    Global   = 1,
    Texture  = 2,
    Shared   = 3,
    Constant = 4,
};

/// Pointer type.
class PtrType : public VectorType {
private:
    PtrType(World& world, const Type* referenced_type, size_t length, int32_t device, AddressSpace addr_space)
        : VectorType(world, Node_PtrType, {referenced_type}, length)
        , addr_space_(addr_space)
        , device_(device)
    {}

public:
    const Type* referenced_type() const { return arg(0); }
    AddressSpace addr_space() const { return addr_space_; }
    int32_t device() const { return device_; }
    bool is_host_device() const { return device_ == -1; }

    virtual uint64_t hash() const override;
    virtual bool equal(const Type* other) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    AddressSpace addr_space_;
    int32_t device_;

    friend class World;
};

/**
 * @brief A struct abstraction.
 *
 * Structs may be recursive via a pointer indirection (like in C or Java).
 * But unlike C, structs may be polymorphic.
 * A concrete instantiation of a struct abstraction is a struct application.
 * @see StructAppType
 */
class StructAbsType : public Type {
private:
    StructAbsType(World& world, size_t size, const std::string& name)
        : Type(world, Node_StructAbsType, Array<const Type*>(size))
        , name_(name)
    {}

public:
    const std::string& name() const { return name_; }
    void set(size_t i, const Type* type) const { const_cast<StructAbsType*>(this)->Type::set(i, type); }
    virtual uint64_t hash() const override { return hash_value(this->gid()); }
    virtual bool equal(const Type* other) const override { return this == other; }
    virtual const Type* instantiate(Types args) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override { THORIN_UNREACHABLE; }

    std::string name_;

    friend class World;
};

/**
 * @brief A struct application.
 *
 * A concrete instantiation of a struct abstraction is a struct application.
 * @see StructAbsType.
 */
class StructAppType : public Type {
private:
    StructAppType(const StructAbsType* struct_abs_type, Types args)
        : Type(struct_abs_type->world(), Node_StructAppType, Array<const Type*>(args.size() + 1))
        , struct_abs_type_(struct_abs_type)
        , elem_cache_(struct_abs_type->num_args())
    {
        set(0, struct_abs_type);
        for (size_t i = 0, e = args.size(); i != e; ++i)
            set(i+1, args[i]);
    }

public:
    const StructAbsType* struct_abs_type() const { return arg(0)->as<StructAbsType>(); }
    Types type_args() const { return args().skip_front(); }
    const Type* type_arg(size_t i) const { return type_args()[i]; }
    size_t num_type_args() const { return type_args().size(); }
    const Type* elem(const Def* def) const { return Type::elem(def); }
    virtual const Type* elem(size_t i) const override;
    Types elems() const;
    size_t num_elems() const { return struct_abs_type()->num_args(); }
    virtual bool use_lea() const override { return true; }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    const StructAbsType* struct_abs_type_;
    mutable Array<const Type*> elem_cache_;

    friend class World;
};

class TupleType : public Type {
private:
    TupleType(World& world, Types args)
        : Type(world, Node_TupleType, args)
    {}

    virtual const Type* vinstantiate(Type2Type&) const override;
    virtual const Type* vrebuild(World& to, Types args) const override;

    friend class World;

public:
    virtual std::ostream& stream(std::ostream&) const override;
};

class FnType : public Type {
private:
    FnType(World& world, Types args)
        : Type(world, Node_FnType, args)
    {
        ++order_;
    }

public:
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class World;
};

class ArrayType : public Type {
protected:
    ArrayType(World& world, NodeKind kind, const Type* elem_type)
        : Type(world, kind, {elem_type})
    {}

public:
    const Type* elem_type() const { return arg(0); }
    virtual bool use_lea() const override { return true; }
};

class IndefiniteArrayType : public ArrayType {
public:
    IndefiniteArrayType(World& world, const Type* elem_type)
        : ArrayType(world, Node_IndefiniteArrayType, elem_type)
    {}

    virtual const IndefiniteArrayType* is_indefinite() const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class World;
};

class DefiniteArrayType : public ArrayType {
public:
    DefiniteArrayType(World& world, const Type* elem_type, u64 dim)
        : ArrayType(world, Node_DefiniteArrayType, elem_type)
        , dim_(dim)
    {}

    u64 dim() const { return dim_; }
    virtual uint64_t hash() const override { return hash_combine(Type::hash(), dim()); }
    virtual bool equal(const Type* other) const override {
        return Type::equal(other) && this->dim() == other->as<DefiniteArrayType>()->dim();
    }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    u64 dim_;

    friend class World;
};

class TypeParam : public Type {
private:
    TypeParam(World& world)
        : Type(world, Node_TypeParam, {})
        , equiv_(nullptr)
    {}

public:
    const Type* bound_at() const { return bound_at_; }
    virtual bool equal(const Type*) const override;
    virtual bool is_closed() const override { return bound_at_ != nullptr; }
    virtual bool is_concrete() const override { return false; }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    mutable const Type* bound_at_;
    mutable const TypeParam* equiv_;
    mutable std::string name_;

    friend bool Type::equal(const Type*) const;
    friend void Type::bind(const TypeParam* type_param) const;
    friend const Type* Type::unify() const;
    friend class World;
};

//------------------------------------------------------------------------------

std::ostream& stream_type_params(std::ostream& os, const Type* type);

//------------------------------------------------------------------------------

}

#endif
