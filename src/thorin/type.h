#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/enums.h"
#include "thorin/util/type_table.h"
#include "thorin/util/symbol.h"

namespace thorin {

//------------------------------------------------------------------------------

class TypeTable;
using Type = TypeBase<TypeTable>;

template<class T>
using TypeMap   = GIDMap<const Type*, T>;
using TypeSet   = GIDSet<const Type*>;
using Type2Type = TypeMap<const Type*>;
using Types     = ArrayRef<const Type*>;

//------------------------------------------------------------------------------

/// Type abstraction.
class Lambda : public Type {
private:
    Lambda(TypeTable& table, const Type* body, const char* name)
        : Type(table, Node_Lambda, {body})
        , name_(name)
    {}

public:
    const char* name() const { return name_; }
    const Type* body() const { return op(0); }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    const char* name_;

    friend class TypeTable;
};

/// Type variable.
class Var : public Type {
private:
    Var(TypeTable& table, int depth)
        : Type(table, Node_Var, {})
        , depth_(depth)
    {
        monomorphic_ = false;
    }

public:
    int depth() const { return depth_; }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override;
    virtual bool equal(const Type*) const override;
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

    int depth_;

    friend class TypeTable;
};

/// Type application.
class App : public Type {
private:
    App(TypeTable& table, const Type* callee, const Type* arg)
        : Type(table, Node_App, {callee, arg})
    {}

public:
    const Type* callee() const { return Type::op(0); }
    const Type* arg() const { return Type::op(1); }
    virtual std::ostream& stream(std::ostream&) const override;
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

private:
    mutable const Type* cache_ = nullptr;
    friend class TypeTable;
};

/// Type of a tuple (structurally typed).
class TupleType : public Type {
private:
    TupleType(TypeTable& table, Types ops)
        : Type(table, Node_TupleType, ops)
    {}

public:
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;
    virtual std::ostream& stream(std::ostream&) const override;

    friend class TypeTable;
};

/// Base class for nominal types (types that have
/// a name that uniquely identifies them).
class NominalType : public Type {
protected:
    NominalType(TypeTable& table, int tag, Symbol name, size_t size)
        : Type(table, tag, thorin::Array<const Type*>(size))
        , name_(name)
    {
        nominal_ = true;
    }

    Symbol name_;

private:
    virtual const Type* vrebuild(TypeTable&, Types) const override;
    virtual const Type* vreduce(int, const Type*, Type2Type&) const override;

public:
    Symbol name() const { return name_; }
    void set(size_t i, const Type* type) const {
        return const_cast<NominalType*>(this)->Type::set(i, type);
    }

    /// Recreates a fresh new nominal type of the
    /// same kind with the same number of operands,
    /// initially all unset.
    virtual const NominalType* stub(TypeTable&) const = 0;
};

class StructType : public NominalType {
private:
    StructType(TypeTable& table, Symbol name, size_t size)
        : NominalType(table, Node_StructType, name, size), field_names_(size)
    {
        // Front-end can and should replace those
        for (size_t i = 0; i < size; i++)
            field_names_[i] = Symbol("field" + std::to_string(i));
    }

    Array<Symbol> field_names_;

    virtual std::ostream& stream(std::ostream&) const override;

public:
    virtual const NominalType* stub(TypeTable&) const override;
    const Symbol& field_name(size_t i) const {
        return field_names_[i];
    }
    void set_field_name(size_t i, Symbol s) const {
        const_cast<StructType*>(this)->field_names_[i] = s;
    }

    friend class TypeTable;
};

class VariantType : public NominalType {
private:
    VariantType(TypeTable& table, Symbol name, size_t size)
        : NominalType(table, Node_VariantType, name, size), variant_names_(size)
    {
        // Front-end can and should replace those
        for (size_t i = 0; i < size; i++)
            variant_names_[i] = Symbol("variant_case" + std::to_string(i));
    }

    Array<Symbol> variant_names_;

    virtual std::ostream& stream(std::ostream&) const override;

public:
    virtual const NominalType* stub(TypeTable&) const override;
    const Symbol& variant_name(size_t i) const {
        return variant_names_[i];
    }
    void set_variant_name(size_t i, Symbol s) const {
        const_cast<VariantType*>(this)->variant_names_[i] = s;
    }

    friend class TypeTable;
};

/// The type of the memory monad.
class MemType : public Type {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    MemType(TypeTable& table)
        : Type(table, Node_MemType, {})
    {}

    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

/// The type of a stack frame.
class FrameType : public Type {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    FrameType(TypeTable& table)
        : Type(table, Node_FrameType, {})
    {}

    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

/// Base class for all SIMD types.
class VectorType : public Type {
protected:
    VectorType(TypeTable& table, int tag, Types ops, size_t length)
        : Type(table, tag, ops)
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
    PrimType(TypeTable& table, PrimTypeTag tag, size_t length)
        : VectorType(table, (int) tag, {}, length)
    {}

public:
    PrimTypeTag primtype_tag() const { return (PrimTypeTag) tag(); }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
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
inline bool is_type_unit(const Type* t) { auto tuple = t->isa<TupleType>(); return tuple && tuple->num_ops() == 0; }

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
    PtrType(TypeTable& table, const Type* pointee, size_t length, int32_t device, AddrSpace addr_space)
        : VectorType(table, Node_PtrType, {pointee}, length)
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
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

    AddrSpace addr_space_;
    int32_t device_;

    friend class TypeTable;
};

/// Returns true if the given type is small enough to fit in a closure environment
inline bool is_thin(const Type* type) {
    return type->isa<PrimType>() || type->isa<PtrType>() || is_type_unit(type);
}

class FnType : public Type {
protected:
    FnType(TypeTable& table, Types ops, int tag = Node_FnType)
        : Type(table, tag, ops)
    {
        ++order_;
    }

public:
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

class ClosureType : public FnType {
private:
    ClosureType(TypeTable& table, Types ops)
        : FnType(table, ops, Node_ClosureType)
    {
        inner_order_ = order_;
        order_ = 0;
    }

public:
    int inner_order() const { return inner_order_; }

    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;
    virtual std::ostream& stream(std::ostream&) const override;

private:
    int inner_order_;

    friend class TypeTable;
};

//------------------------------------------------------------------------------

class ArrayType : public Type {
protected:
    ArrayType(TypeTable& table, int tag, const Type* elem_type)
        : Type(table, tag, {elem_type})
    {}

public:
    const Type* elem_type() const { return op(0); }
};

class IndefiniteArrayType : public ArrayType {
public:
    IndefiniteArrayType(TypeTable& table, const Type* elem_type)
        : ArrayType(table, Node_IndefiniteArrayType, elem_type)
    {}

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

class DefiniteArrayType : public ArrayType {
public:
    DefiniteArrayType(TypeTable& table, const Type* elem_type, u64 dim)
        : ArrayType(table, Node_DefiniteArrayType, elem_type)
        , dim_(dim)
    {}

    u64 dim() const { return dim_; }
    virtual uint64_t vhash() const override { return hash_combine(Type::vhash(), dim()); }
    virtual bool equal(const Type* other) const override {
        return Type::equal(other) && this->dim() == other->as<DefiniteArrayType>()->dim();
    }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Type* vrebuild(TypeTable& to, Types ops) const override;

    u64 dim_;

    friend class TypeTable;
};

bool use_lea(const Type*);

//------------------------------------------------------------------------------

/// Container for all types. Types are hashed and can be compared using pointer equality.
class TypeTable : public TypeTableBase<Type> {
public:
    TypeTable();

    const Var* var(int depth) { return unify(new Var(*this, depth)); }
    const Lambda* lambda(const Type* body, const char* name) { return unify(new Lambda(*this, body, name)); }
    const Type* app(const Type* callee, const Type* arg);

    const Type* tuple_type(Types ops) { return ops.size() == 1 ? ops.front() : unify(new TupleType(*this, ops)); }
    const TupleType* unit() { return unit_; } ///< Returns unit, i.e., an empty @p TupleType.
    const VariantType* variant_type(Symbol name, size_t size);
    const StructType* struct_type(Symbol name, size_t size);

#define THORIN_ALL_TYPE(T, M) \
    const PrimType* type_##T(size_t length = 1) { return type(PrimType_##T, length); }
#include "thorin/tables/primtypetable.h"
    const PrimType* type(PrimTypeTag tag, size_t length = 1) {
        size_t i = tag - Begin_PrimType;
        assert(i < (size_t) Num_PrimTypes);
        return length == 1 ? primtypes_[i] : unify(new PrimType(*this, tag, length));
    }
    const MemType* mem_type() const { return mem_; }
    const FrameType* frame_type() const { return frame_; }
    const PtrType* ptr_type(const Type* pointee,
                            size_t length = 1, int32_t device = -1, AddrSpace addr_space = AddrSpace::Generic) {
        return unify(new PtrType(*this, pointee, length, device, addr_space));
    }
    const FnType* fn_type() { return fn0_; } ///< Returns an empty @p FnType.
    const FnType* fn_type(Types args) { return unify(new FnType(*this, args)); }
    const ClosureType* closure_type(Types args) { return unify(new ClosureType(*this, args)); }
    const DefiniteArrayType*   definite_array_type(const Type* elem, u64 dim) { return unify(new DefiniteArrayType(*this, elem, dim)); }
    const IndefiniteArrayType* indefinite_array_type(const Type* elem) { return unify(new IndefiniteArrayType(*this, elem)); }

    friend void swap(TypeTable& t1, TypeTable& t2) {
        using std::swap;
        swap(t1.types_, t2.types_);
        swap(t1.unit_,  t2.unit_);
        swap(t1.fn0_,   t2.fn0_);
        swap(t1.mem_,   t2.mem_);
        swap(t1.frame_, t2.frame_);
#define THORIN_ALL_TYPE(T, M) \
        swap(t1.T##_,   t2.T##_);
#include "thorin/tables/primtypetable.h"

        t1.fix();
        t2.fix();
    }

private:
    void fix() {
        for (auto type : types_)
            type->table_ = this;
    }

private:
    const TupleType* unit_; ///< tuple().
    const FnType* fn0_;
    const MemType* mem_;
    const FrameType* frame_;
    union {
        struct {
#define THORIN_ALL_TYPE(T, M) const PrimType* T##_;
#include "thorin/tables/primtypetable.h"
        };

        const PrimType* primtypes_[Num_PrimTypes];
    };

    friend class Lambda;
};

//------------------------------------------------------------------------------

}

#endif
