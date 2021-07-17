#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/enums.h"
#include "thorin/util/hash.h"
#include "thorin/util/cast.h"
#include "thorin/util/stream.h"
#include "thorin/util/array.h"
#include "thorin/util/symbol.h"

namespace thorin {

class TypeTable;
class Type;

template<class To>
using TypeMap   = GIDMap<const Type*, To>;
using Type2Type = TypeMap<const Type*>;
using TypeSet   = GIDSet<const Type*>;
using Types     = ArrayRef<const Type*>;

/// Base class for all \p Type%s.
class Type : public RuntimeCast<Type>, public Streamable<Type> {
protected:
    Type(TypeTable& table, int tag, Types ops);

    void set(size_t i, const Type* type) {
        ops_[i] = type;
        order_ = std::max(order_, type->order());
    }

public:
    int tag() const { return tag_; }
    TypeTable& table() const { return *table_; }

    Types ops() const { return ops_; }
    const Type* op(size_t i) const { return ops()[i]; }
    size_t num_ops() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }

    bool is_nominal() const { return nominal_; } ///< A nominal @p Type is always different from each other @p Type.
    int order() const { return order_; }
    size_t gid() const { return gid_; }
    hash_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }
    virtual bool equal(const Type*) const;

    const Type* rebuild(TypeTable& to, Types ops) const {
        assert(num_ops() == ops.size());
        if (ops.empty() && &table() == &to)
            return this;
        return vrebuild(to, ops);
    }
    const Type* rebuild(Types ops) const { return rebuild(table(), ops); }
    Stream& stream(Stream&) const;
    void dump() const;

protected:
    virtual hash_t vhash() const;

    mutable hash_t hash_ = 0;
    mutable bool nominal_  = false;
    int order_ = 0;
    size_t gid_;

private:
    virtual const Type* vrebuild(TypeTable& to, Types ops) const = 0;

    mutable TypeTable* table_;

    int tag_;
    thorin::Array<const Type*> ops_;

    friend TypeTable;
};

/// Type of a tuple (structurally typed).
class TupleType : public Type {
private:
    TupleType(TypeTable& table, Types ops)
        : Type(table, Node_TupleType, ops)
    {}

public:
    const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

/// Base class for nominal types (types that have
/// a name that uniquely identifies them).
class NominalType : public Type {
protected:
    NominalType(TypeTable& table, int tag, Symbol name, size_t size, size_t gid)
        : Type(table, tag, thorin::Array<const Type*>(size))
        , name_(name)
        , op_names_(size)
    {
        nominal_ = true;
        gid_ = gid;
    }

    Symbol name_;
    Array<Symbol> op_names_;

private:
    const Type* vrebuild(TypeTable&, Types) const override;

public:
    Symbol name() const { return name_; }
    Symbol op_name(size_t i) const { return op_names_[i]; }
    void set(size_t i, const Type* type) const {
        return const_cast<NominalType*>(this)->Type::set(i, type);
    }
    void set_op_name(size_t i, Symbol name) const {
        const_cast<NominalType*>(this)->op_names_[i] = name;
    }
    Array<Symbol>& op_names() const {
        return const_cast<NominalType*>(this)->op_names_;
    }

    /// Recreates a fresh new nominal type of the
    /// same kind with the same number of operands,
    /// initially all unset.
    virtual const NominalType* stub(TypeTable&) const = 0;
};

class StructType : public NominalType {
private:
    StructType(TypeTable& table, Symbol name, size_t size, size_t gid)
        : NominalType(table, Node_StructType, name, size, gid)
    {}

public:
    const NominalType* stub(TypeTable&) const override;

    friend class TypeTable;
};

class VariantType : public NominalType {
private:
    VariantType(TypeTable& table, Symbol name, size_t size, size_t gid)
        : NominalType(table, Node_VariantType, name, size, gid)
    {}

public:
    const NominalType* stub(TypeTable&) const override;

    bool has_payload() const;

    friend class TypeTable;
};

/// The type of the memory monad.
class MemType : public Type {
private:
    MemType(TypeTable& table)
        : Type(table, Node_MemType, {})
    {}

    const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

/// The type of App nodes.
class BottomType : public Type {
private:
    BottomType(TypeTable& table)
            : Type(table, Node_BotType, {})
    {}

    const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

/// The type of a stack frame.
class FrameType : public Type {
private:
    FrameType(TypeTable& table)
        : Type(table, Node_FrameType, {})
    {}

    const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

/// Base class for all SIMD types.
class VectorType : public Type {
protected:
    VectorType(TypeTable& table, int tag, Types ops, size_t length)
        : Type(table, tag, ops)
        , length_(length)
    {}

    hash_t vhash() const override { return hash_combine(Type::vhash(), length()); }
    bool equal(const Type* other) const override {
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
    const Type* vrebuild(TypeTable& to, Types ops) const override;

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

    hash_t vhash() const override;
    bool equal(const Type* other) const override;

private:
    const Type* vrebuild(TypeTable& to, Types ops) const override;

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

private:
    const Type* vrebuild(TypeTable& to, Types ops) const override;

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
    const Type* vrebuild(TypeTable& to, Types ops) const override;

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

private:
    const Type* vrebuild(TypeTable& to, Types ops) const override;

    friend class TypeTable;
};

class DefiniteArrayType : public ArrayType {
public:
    DefiniteArrayType(TypeTable& table, const Type* elem_type, u64 dim)
        : ArrayType(table, Node_DefiniteArrayType, elem_type)
        , dim_(dim)
    {}

    u64 dim() const { return dim_; }
    hash_t vhash() const override { return hash_combine(Type::vhash(), dim()); }
    bool equal(const Type* other) const override {
        return Type::equal(other) && this->dim() == other->as<DefiniteArrayType>()->dim();
    }

private:
    const Type* vrebuild(TypeTable& to, Types ops) const override;

    u64 dim_;

    friend class TypeTable;
};

bool use_lea(const Type*);

//------------------------------------------------------------------------------

/// Container for all types. Types are hashed and can be compared using pointer equality.
class TypeTable {
private:
    struct TypeHash {
        static hash_t hash(const Type* t) { return t->hash(); }
        static bool eq(const Type* t1, const Type* t2) { return t2->equal(t1); }
        static const Type* sentinel() { return (const Type*)(1); }
    };

    typedef thorin::HashSet<const Type*, TypeHash> TypeSet;

public:
    TypeTable();

    const Type* tuple_type(Types ops);
    const TupleType* unit() { return unit_; } ///< Returns unit, i.e., an empty @p TupleType.
    const VariantType* variant_type(Symbol name, size_t size);
    const StructType* struct_type(Symbol name, size_t size);

#define THORIN_ALL_TYPE(T, M) \
    const PrimType* type_##T(size_t length = 1) { return prim_type(PrimType_##T, length); }
#include "thorin/tables/primtypetable.h"
    const PrimType* prim_type(PrimTypeTag tag, size_t length = 1);
    const BottomType* bottom_type() const { return bottom_ty_; }
    const MemType* mem_type() const { return mem_; }
    const FrameType* frame_type() const { return frame_; }
    const PtrType* ptr_type(const Type* pointee, size_t length = 1, int32_t device = -1, AddrSpace addr_space = AddrSpace::Generic);
    const FnType* fn_type() { return fn0_; } ///< Returns an empty @p FnType.
    const FnType* fn_type(Types args);
    const ClosureType* closure_type(Types args);
    const DefiniteArrayType*   definite_array_type(const Type* elem, u64 dim);
    const IndefiniteArrayType* indefinite_array_type(const Type* elem);

    const TypeSet& types() const { return types_; }

    friend void swap(TypeTable& t1, TypeTable& t2) {
        using std::swap;
        swap(t1.types_, t2.types_);
        swap(t1.unit_,  t2.unit_);
        swap(t1.fn0_,   t2.fn0_);
        swap(t1.bottom_ty_,   t2.bottom_ty_);
        swap(t1.mem_,   t2.mem_);
        swap(t1.frame_, t2.frame_);
        std::swap_ranges(t1.primtypes_, t1.primtypes_ + Num_PrimTypes, t2.primtypes_);

        t1.fix();
        t2.fix();
    }

private:
    void fix() {
        for (auto type : types_)
            type->table_ = this;
    }

    template <typename T, typename... Args>
    const T* insert(Args&&... args);

private:
    TypeSet types_;

    const TupleType* unit_; ///< tuple().
    const FnType* fn0_;
    const BottomType* bottom_ty_;
    const MemType* mem_;
    const FrameType* frame_;
    const PrimType* primtypes_[Num_PrimTypes];
};

//------------------------------------------------------------------------------

}

#endif
