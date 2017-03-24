#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"
#include "thorin/util/log.h"

namespace thorin {

#define HENK_STRUCT_EXTRA_NAME name
#define HENK_STRUCT_EXTRA_TYPE const char*
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

    virtual const Type* vrebuild(World& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

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

    virtual const Type* vrebuild(World& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    friend class World;
};

/// Base class for all SIMD types.
class VectorType : public Type {
protected:
    VectorType(World& world, int tag, Types ops, size_t length)
        : Type(world, tag, ops)
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
    PrimType(World& world, PrimTypeTag tag, size_t length)
        : VectorType(world, (int) tag, {}, length)
    {}

public:
    PrimTypeTag primtype_tag() const { return (PrimTypeTag) tag(); }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    friend class World;
};

inline bool is_primtype (const Type* t) { return thorin::is_primtype(t->tag()); }
inline bool is_type_ps  (const Type* t) { return thorin::is_type_ps (t->tag()); }
inline bool is_type_pu  (const Type* t) { return thorin::is_type_pu (t->tag()); }
inline bool is_type_qs  (const Type* t) { return thorin::is_type_qs (t->tag()); }
inline bool is_type_qu  (const Type* t) { return thorin::is_type_qu (t->tag()); }
inline bool is_type_pf  (const Type* t) { return thorin::is_type_pf (t->tag()); }
inline bool is_type_qf  (const Type* t) { return thorin::is_type_qf (t->tag()); }
inline bool is_type_p   (const Type* t) { return thorin::is_type_p  (t->tag()); }
inline bool is_type_q   (const Type* t) { return thorin::is_type_q  (t->tag()); }
inline bool is_type_s   (const Type* t) { return thorin::is_type_s  (t->tag()); }
inline bool is_type_u   (const Type* t) { return thorin::is_type_u  (t->tag()); }
inline bool is_type_i   (const Type* t) { return thorin::is_type_i  (t->tag()); }
inline bool is_type_f   (const Type* t) { return thorin::is_type_f  (t->tag()); }
inline bool is_type_bool(const Type* t) { return t->tag() == Node_PrimType_bool; }

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
    PtrType(World& world, const Type* pointee, size_t length, int32_t device, AddrSpace addr_space)
        : VectorType(world, Node_PtrType, {pointee}, length)
        , addr_space_(addr_space)
        , device_(device)
    {}

public:
    const Type* pointee() const { return op(0); }
    AddrSpace addr_space() const { return addr_space_; }
    int32_t device() const { return device_; }
    bool is_host_device() const { return device_ == -1; }

    virtual uint64_t vhash() const override;
    virtual bool equal(const Type* other) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    AddrSpace addr_space_;
    int32_t device_;

    friend class World;
};

class FnType : public Type {
private:
    FnType(World& world, Types ops)
        : Type(world, Node_FnType, ops)
    {
        ++order_;
    }

public:
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    friend class World;
};

//------------------------------------------------------------------------------

class ArrayType : public Type {
protected:
    ArrayType(World& world, int tag, const Type* elem_type)
        : Type(world, tag, {elem_type})
    {}

public:
    const Type* elem_type() const { return op(0); }
};

class IndefiniteArrayType : public ArrayType {
public:
    IndefiniteArrayType(World& world, const Type* elem_type)
        : ArrayType(world, Node_IndefiniteArrayType, elem_type)
    {}

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(World& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

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
    virtual const Type* vrebuild(World& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    u64 dim_;

    friend class World;
};

bool use_lea(const Type*);

//------------------------------------------------------------------------------

}

#endif
