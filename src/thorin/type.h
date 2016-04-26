#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"

namespace thorin {

#define HENK_TABLE_NAME world
#define HENK_TABLE_TYPE World
#include "thorin/henk.h"

//------------------------------------------------------------------------------

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
    VectorType(World& world, int kind, Types args, size_t length)
        : Type(world, kind, args)
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

/// Returns the vector length. Raises an assertion if this type is not a @p VectorType.
inline size_t vector_length(const Type* type) { return type->as<VectorType>()->length(); }

/// Primitive type.
class PrimType : public VectorType {
private:
    PrimType(World& world, PrimTypeKind kind, size_t length)
        : VectorType(world, (int) kind, {}, length)
    {}

public:
    PrimTypeKind primtype_kind() const { return (PrimTypeKind) kind(); }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    friend class World;
};

inline bool is_primtype (const Type* t) { return thorin::is_primtype(t->kind()); }
inline bool is_type_ps  (const Type* t) { return thorin::is_type_ps (t->kind()); }
inline bool is_type_pu  (const Type* t) { return thorin::is_type_pu (t->kind()); }
inline bool is_type_qs  (const Type* t) { return thorin::is_type_qs (t->kind()); }
inline bool is_type_qu  (const Type* t) { return thorin::is_type_qu (t->kind()); }
inline bool is_type_pf  (const Type* t) { return thorin::is_type_pf (t->kind()); }
inline bool is_type_qf  (const Type* t) { return thorin::is_type_qf (t->kind()); }
inline bool is_type_p   (const Type* t) { return thorin::is_type_p  (t->kind()); }
inline bool is_type_q   (const Type* t) { return thorin::is_type_q  (t->kind()); }
inline bool is_type_s   (const Type* t) { return thorin::is_type_s  (t->kind()); }
inline bool is_type_u   (const Type* t) { return thorin::is_type_u  (t->kind()); }
inline bool is_type_i   (const Type* t) { return thorin::is_type_i  (t->kind()); }
inline bool is_type_f   (const Type* t) { return thorin::is_type_f  (t->kind()); }
inline bool is_type_bool(const Type* t) { return t->kind() == Node_PrimType_bool; }

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
    PtrType(World& world, const Type* referenced_type, size_t length, int32_t device, AddrSpace addr_space)
        : VectorType(world, Node_PtrType, {referenced_type}, length)
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
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    AddrSpace addr_space_;
    int32_t device_;

    friend class World;
};

/**
 * A struct abstraction.
 * Structs may be recursive via a pointer indirection (like in C or Java).
 * But unlike C, structs may be polymorphic.
 * A concrete instantiation of a struct abstraction is a struct application.
 * @see StructAppType
 */
class StructAbsType : public Type {
private:
    StructAbsType(World& world, size_t size, size_t num_type_params, const std::string& name)
        : Type(world, Node_StructAbsType, Array<const Type*>(size), num_type_params)
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
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override { THORIN_UNREACHABLE; }

    std::string name_;

    friend class World;
};

/**
 * A struct application.
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

    Types elems() const;
    const Type* elem(size_t i) const;
    size_t num_elems() const { return struct_abs_type()->num_args(); }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types args) const override;
    virtual const Type* vinstantiate(Type2Type&) const override;

    const StructAbsType* struct_abs_type_;
    mutable Array<const Type*> elem_cache_;

    friend class World;
};

class FnType : public Type {
private:
    FnType(World& world, Types args, size_t num_type_params)
        : Type(world, Node_FnType, args, num_type_params)
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

//------------------------------------------------------------------------------

class ArrayType : public Type {
protected:
    ArrayType(World& world, int kind, const Type* elem_type)
        : Type(world, kind, {elem_type})
    {}

public:
    const Type* elem_type() const { return arg(0); }
};

class IndefiniteArrayType : public ArrayType {
public:
    IndefiniteArrayType(World& world, const Type* elem_type)
        : ArrayType(world, Node_IndefiniteArrayType, elem_type)
    {}

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
    virtual uint64_t vhash() const override { return hash_combine(Type::vhash(), dim()); }
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

const IndefiniteArrayType* is_indefinite(const Type*);
bool use_lea(const Type*);

//------------------------------------------------------------------------------

}

#endif
