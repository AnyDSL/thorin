#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"

namespace thorin {

//------------------------------------------------------------------------------

class Def;
class IndefiniteArrayType;
class Type;
class TypeParam;
class TypeTable;

template<class T>
struct GIDHash {
    uint64_t operator()(T n) const { return n->gid(); }
};

template<class Key, class Value>
using GIDMap    = HashMap<const Key*, Value, GIDHash<const Key*>>;
template<class Key>
using GIDSet    = HashSet<const Key*, GIDHash<const Key*>>;

template<class To>
using TypeMap      = GIDMap<Type, To>;
using TypeSet      = GIDSet<Type>;
using Type2Type    = TypeMap<const Type*>;

typedef ArrayRef<const Type*> Types;

//------------------------------------------------------------------------------

/// Base class for all \p Type%s.
class Type : public Streamable, public MagicCast<Type> {
protected:
    Type(const Type&) = delete;
    Type& operator=(const Type&) = delete;

    Type(TypeTable& typetable, NodeKind kind, Types args, size_t num_type_params = 0)
        : typetable_(typetable)
        , kind_(kind)
        , args_(args.size())
        , type_params_(num_type_params)
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
        closed_ &= type->is_closed();
        monomorphic_ &= type->is_monomorphic();
    }

public:
    NodeKind kind() const { return kind_; }
    Types args() const { return args_; }
    ArrayRef<const TypeParam*> type_params() const { return type_params_; }
    size_t num_type_params() const { return type_params().size(); }
    const Type* arg(size_t i) const { assert(i < args().size()); return args()[i]; }
    const TypeParam* type_param(size_t i) const { assert(i < type_params().size()); return type_params()[i]; }
    size_t num_args() const { return args_.size(); }
    bool is_hashed() const { return hashed_; }          ///< This @p Type is already recorded inside of @p TypeTable.
    bool empty() const { return args_.empty(); }
    TypeTable& typetable() const { return typetable_; }
    size_t gid() const { return gid_; }
    int order() const { return order_; }
    bool is_closed() const { return closed_; }  ///< Are all @p TypeParam%s bound?
    bool is_monomorphic() const { return monomorphic_; }        ///< Does this @p Type not depend on any @p TypeParam%s?.
    bool is_polymorphic() const { return !is_monomorphic(); }   ///< Does this @p Type depend on any @p TypeParam%s?.
    size_t length() const; ///< Returns the vector length. Raises an assertion if this type is not a @p VectorType.
    virtual const Type* instantiate(Types) const;
    const Type* instantiate(Type2Type&) const;
    const Type* specialize(Type2Type&) const;
    const Type* elem(const Def*) const;
    const Type* rebuild(TypeTable& to, Types args) const;
    const Type* rebuild(Types args) const { return rebuild(typetable(), args); }
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

    uint64_t hash() const { return is_hashed() ? hash_ : hash_ = vhash(); }
    virtual uint64_t vhash() const;
    virtual bool equal(const Type*) const;
    virtual const IndefiniteArrayType* is_indefinite() const;
    virtual bool use_lea() const { return false; }
    virtual const Type* vinstantiate(Type2Type&) const = 0;

    static size_t gid_counter() { return gid_counter_; }

protected:
    Array<const Type*> specialize_args(Type2Type&) const;

    int order_ = 0;
    mutable uint64_t hash_ = 0;
    mutable bool hashed_ = false;
    mutable bool closed_ = true;
    mutable bool monomorphic_ = true;

private:
    virtual const Type* vrebuild(TypeTable& to, Types args) const = 0;

    TypeTable& typetable_;
    NodeKind kind_;
    Array<const Type*> args_;
    mutable Array<const TypeParam*> type_params_;
    mutable size_t gid_;
    static size_t gid_counter_;

    friend const Type* close_base(const Type*&, ArrayRef<const TypeParam*>);
    template<class> friend class TypeTableBase;
};

template<class T>
const T* close(const T*& type, ArrayRef<const TypeParam*> type_param) {
    static_assert(std::is_base_of<Type, T>::value, "T is not a base of thorin::Type");
    return close_base((const Type*&) type, type_param)->template as<T>();
}

/// The type of the memory monad.
class MemType : public Type {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    MemType(TypeTable& typetable)
        : Type(typetable, Node_MemType, {})
    {}

    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class TypeTable;
};

/// The type of a stack frame.
class FrameType : public Type {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    FrameType(TypeTable& typetable)
        : Type(typetable, Node_FrameType, {})
    {}

    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class TypeTable;
};

/// Base class for all SIMD types.
class VectorType : public Type {
protected:
    VectorType(TypeTable& typetable, NodeKind kind, Types args, size_t length)
        : Type(typetable, kind, args)
        , length_(length)
    {}

    virtual uint64_t vhash() const override { return hash_combine(Type::vhash(), length()); }
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
    PrimType(TypeTable& typetable, PrimTypeKind kind, size_t length)
        : VectorType(typetable, (NodeKind) kind, {}, length)
    {}

public:
    PrimTypeKind primtype_kind() const { return (PrimTypeKind) kind(); }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class TypeTable;
};

enum class AddrSpace : uint32_t {
    Generic  = 0,
    Global   = 1,
    Texture  = 2,
    Shared   = 3,
    Constant = 4,
};

/// Pointer type.
class PtrType : public VectorType {
private:
    PtrType(TypeTable& typetable, const Type* referenced_type, size_t length, int32_t device, AddrSpace addr_space)
        : VectorType(typetable, Node_PtrType, {referenced_type}, length)
        , addr_space_(addr_space)
        , device_(device)
    {}

public:
    const Type* referenced_type() const { return arg(0); }
    AddrSpace addr_space() const { return addr_space_; }
    int32_t device() const { return device_; }
    bool is_host_device() const { return device_ == -1; }

    virtual uint64_t vhash() const override;
    virtual bool equal(const Type* other) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    AddrSpace addr_space_;
    int32_t device_;

    friend class TypeTable;
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
    StructAbsType(TypeTable& typetable, size_t size, size_t num_type_params, const std::string& name)
        : Type(typetable, Node_StructAbsType, Array<const Type*>(size), num_type_params)
        , name_(name)
    {}

public:
    const std::string& name() const { return name_; }
    void set(size_t i, const Type* type) const { const_cast<StructAbsType*>(this)->Type::set(i, type); }
    virtual uint64_t vhash() const override { return hash_value(this->gid()); }
    virtual bool equal(const Type* other) const override { return this == other; }
    virtual const Type* instantiate(Types args) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override { THORIN_UNREACHABLE; }

    std::string name_;

    friend class TypeTable;
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
        : Type(struct_abs_type->typetable(), Node_StructAppType, Array<const Type*>(args.size() + 1))
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
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    const StructAbsType* struct_abs_type_;
    mutable Array<const Type*> elem_cache_;

    friend class TypeTable;
};

class TupleType : public Type {
private:
    TupleType(TypeTable& typetable, Types args)
        : Type(typetable, Node_TupleType, args)
    {}

    virtual const Type* vinstantiate(Type2Type&) const override;
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;

public:
    virtual std::ostream& stream(std::ostream&) const override;

    template<class> friend class TypeTableBase;
};

class FnType : public Type {
private:
    FnType(TypeTable& typetable, Types args, size_t num_type_params)
        : Type(typetable, Node_FnType, args, num_type_params)
    {
        ++order_;
    }

public:
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class TypeTable;
};

class ArrayType : public Type {
protected:
    ArrayType(TypeTable& typetable, NodeKind kind, const Type* elem_type)
        : Type(typetable, kind, {elem_type})
    {}

public:
    const Type* elem_type() const { return arg(0); }
    virtual bool use_lea() const override { return true; }
};

class IndefiniteArrayType : public ArrayType {
public:
    IndefiniteArrayType(TypeTable& typetable, const Type* elem_type)
        : ArrayType(typetable, Node_IndefiniteArrayType, elem_type)
    {}

    virtual const IndefiniteArrayType* is_indefinite() const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class TypeTable;
};

class DefiniteArrayType : public ArrayType {
public:
    DefiniteArrayType(TypeTable& typetable, const Type* elem_type, u64 dim)
        : ArrayType(typetable, Node_DefiniteArrayType, elem_type)
        , dim_(dim)
    {}

    u64 dim() const { return dim_; }
    virtual uint64_t vhash() const override { return hash_combine(Type::vhash(), dim()); }
    virtual bool equal(const Type* other) const override {
        return Type::equal(other) && this->dim() == other->as<DefiniteArrayType>()->dim();
    }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    u64 dim_;

    friend class TypeTable;
};

class TypeParam : public Type {
private:
    TypeParam(TypeTable& typetable, const std::string& name)
        : Type(typetable, Node_TypeParam, {})
        , name_(name)
    {
        closed_ = false;
        monomorphic_ = false;
    }

public:
    const std::string& name() const { return name_; }
    const Type* binder() const { return binder_; }
    size_t index() const { return index_; }
    virtual bool equal(const Type*) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override;
    virtual const Type* vrebuild(TypeTable& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    std::string name_;
    mutable const Type* binder_;
    mutable size_t index_;
    mutable const TypeParam* equiv_ = nullptr;

    friend bool Type::equal(const Type*) const;
    friend const Type* close_base(const Type*&, ArrayRef<const TypeParam*>);
    template<class> friend class TypeTableBase;
};

std::ostream& stream_type_params(std::ostream& os, const Type* type);

//------------------------------------------------------------------------------

}

#endif
