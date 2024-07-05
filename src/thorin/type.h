#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/def.h"
#include "thorin/enums.h"
#include "thorin/util/hash.h"
#include "thorin/util/cast.h"
#include "thorin/util/stream.h"
#include "thorin/util/array.h"
#include "thorin/util/symbol.h"

namespace thorin {

class TypeTable;
class Type;
using Types = ArrayRef<const Type*>;

/// Base class for all \p Type%s.
class Type : public Def {
protected:
    /// Constructor for a @em structural Type.
    Type(World& w, NodeTag tag, const Type* type, Defs args, Debug dbg) : Def(w, tag, type, args, dbg) {}
    Type(World& w, NodeTag tag, Defs args, Debug dbg);
    /// Constructor for a @em nom Type.
    Type(World& w, NodeTag tag, const Type* type, size_t size, Debug dbg) : Def(w, tag, type, size, dbg) {}
    Type(World& w, NodeTag tag, size_t size, Debug dbg);

public:
    int order() const override { return order_; }
    void set_op(size_t i, const Def *def) override;
    Stream& stream(Stream&) const;

    std::vector<const Type*> filter_type_ops() const {
        std::vector<const Type*> type_ops;
        for (auto& op : ops()) {
            if (auto t = op->isa<Type>())
                type_ops.push_back(t);
        }
        return type_ops;
    }
protected:
    int order_ = 0;
    friend class World;
};

class Star : public Type {
protected:
    explicit Star(World& w) : Type(w, Node_Star, nullptr, 0, {}) {
        set_type(this);
    }

    friend class World;
};

Array<const Def*> types2defs(ArrayRef<const Type*> types);
Array<const Type*> defs2types(ArrayRef<const Def*> types);

template<class T>
class TypeOpsMixin {
public:
    Types types() const {
        Defs defs = static_cast<const T*>(this)->ops();
        const Def* const* ptr = defs.begin();
        auto ptr2 = reinterpret_cast<const Type* const*>(ptr);
        auto types = Types(ptr2, defs.size());
        return types;
    }
};

/// Type of a tuple (structurally typed).
class TupleType : public Type, public TypeOpsMixin<TupleType> {
private:
    TupleType(World& world, Defs ops, Debug dbg)
        : Type(world, Node_TupleType, ops, dbg)
    {}

public:
    const Type* rebuild(World&, const Type*, Defs) const override;
    friend class World;
};

/// Base class for nominal types (types that have
/// a name that uniquely identifies them).
class NominalType : public Type {
protected:
    NominalType(World& world, NodeTag tag, Symbol name, size_t size, Debug dbg)
        : Type(world, tag, size, dbg)
        , name_(name)
        , op_names_(size)
    {}

    Symbol name_;
    Array<Symbol> op_names_;

private:
    const Type* rebuild(World&, const Type*, Defs) const override;

public:
    Symbol name() const { return name_; }
    using Type::op_name; //Would be hidden otherwise.
    Symbol op_name(size_t i) const { return op_names_[i]; }
    void set_op_name(size_t i, Symbol name) const {
        const_cast<NominalType*>(this)->op_names_[i] = name;
    }
    Array<Symbol>& op_names() const {
        return const_cast<NominalType*>(this)->op_names_;
    }
};

class StructType : public NominalType, public TypeOpsMixin<StructType> {
private:
    StructType(World& world, Symbol name, size_t size, Debug dbg)
        : NominalType(world, Node_StructType, name, size, dbg)
    {}

public:
    virtual StructType* stub(Rewriter&, const Type*) const override;

    friend class World;
};

class VariantType : public NominalType, public TypeOpsMixin<VariantType> {
private:
    VariantType(World& world, Symbol name, size_t size, Debug dbg)
        : NominalType(world, Node_VariantType, name, size, dbg)
    {}

public:
    virtual VariantType* stub(Rewriter&, const Type*) const override;

    bool has_payload() const;

    friend class World;
};

/// The type of the memory monad.
class MemType : public Type {
private:
    MemType(World& world, Debug dbg)
        : Type(world, Node_MemType, Defs(), dbg)
    {}

    const Type* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// The type of App nodes.
class BottomType : public Type {
private:
    BottomType(World& world, Debug dbg)
        : Type(world, Node_BotType, Defs(), dbg)
    {}

    const Type* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// The type of a stack frame.
class FrameType : public Type {
private:
    FrameType(World& world, Debug dbg)
        : Type(world, Node_FrameType, Defs(), dbg)
    {}

    const Type* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

/// Base class for all SIMD types.
class VectorType : public Type {
protected:
    VectorType(World& world, NodeTag tag, Defs ops, size_t length, Debug dbg)
        : Type(world, tag, ops, dbg)
        , length_(length)
    {}

    hash_t vhash() const override { return hash_combine(Type::vhash(), length()); }
    bool equal(const Def* other) const override {
        return Def::equal(other) && this->length() == other->as<VectorType>()->length();
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
    PrimType(World& world, PrimTypeTag tag, size_t length, Debug dbg)
        : VectorType(world, (NodeTag) tag, Defs(), length, dbg)
    {}

public:
    PrimTypeTag primtype_tag() const { return (PrimTypeTag) tag(); }
    const Type* rebuild(World&, const Type*, Defs) const override;

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
inline bool is_type_unit(const Type* t) { auto tuple = t->isa<TupleType>(); return tuple && tuple->num_ops() == 0; }

enum class AddrSpace : uint32_t {
    Generic  = 0,
    Global   = 1,
    Texture  = 2,
    Shared   = 3,
    Constant = 4,
    Private =  5, // Corresponds to the 'private' storage class in compute kernels/shaders, as in thread-private
    Function = 6, // Corresponds to the 'function' storage class in SPIR-V
    Push     = 7, // Corresponds to the 'push constant' storage class in SPIR-V
    Input    = 8,
    Output   = 9,
};

/// Pointer type.
class PtrType : public VectorType, public TypeOpsMixin<PtrType> {
private:
    PtrType(World& world, const Type* pointee, size_t length, AddrSpace addr_space, Debug dbg)
        : VectorType(world, Node_PtrType, {pointee}, length, dbg)
        , addr_space_(addr_space)
    {}

public:
    const Type* pointee() const { return op(0)->as<Type>(); }
    AddrSpace addr_space() const { return addr_space_; }

    hash_t vhash() const override;
    bool equal(const Def* other) const override;

private:
    const Type* rebuild(World&, const Type*, Defs) const override;

    AddrSpace addr_space_;

    friend class World;
};

/// Returns true if the given type is small enough to fit in a closure environment
inline bool is_thin(const Type* type) {
    return type->isa<PrimType>() || type->isa<PtrType>() || is_type_unit(type);
}

class FnType : public Type, public TypeOpsMixin<FnType> {
protected:
    FnType(World& world, Defs ops, NodeTag tag, Debug dbg)
        : Type(world, tag, ops, dbg)
    {
        ++order_;
    }

public:
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

private:
    const Type* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

class ClosureType : public FnType {
private:
    ClosureType(World& world, Defs ops, Debug dbg)
        : FnType(world, ops, Node_ClosureType, dbg)
    {
        inner_order_ = order_;
        order_ = 0;
    }

public:
    int inner_order() const { return inner_order_; }
    const Type* rebuild(World&, const Type*, Defs) const override;

private:
    int inner_order_;

    friend class World;
};

//------------------------------------------------------------------------------

class ArrayType : public Type, public TypeOpsMixin<ArrayType> {
protected:
    ArrayType(World& world, NodeTag tag, const Type* elem_type, Debug dbg)
        : Type(world, tag, {elem_type}, dbg)
    {}

public:
    const Type* elem_type() const { return op(0)->as<Type>(); }
};

class IndefiniteArrayType : public ArrayType {
public:
    IndefiniteArrayType(World& world, const Type* elem_type, Debug dbg)
        : ArrayType(world, Node_IndefiniteArrayType, elem_type, dbg)
    {}

private:
    const Type* rebuild(World&, const Type*, Defs) const override;

    friend class World;
};

class DefiniteArrayType : public ArrayType {
public:
    DefiniteArrayType(World& world, const Type* elem_type, u64 dim, Debug dbg)
        : ArrayType(world, Node_DefiniteArrayType, elem_type, dbg)
        , dim_(dim)
    {}

    u64 dim() const { return dim_; }
    hash_t vhash() const override { return hash_combine(Type::vhash(), dim()); }
    bool equal(const Def* other) const override {
        return Def::equal(other) && this->dim() == other->as<DefiniteArrayType>()->dim();
    }

private:
    const Type* rebuild(World&, const Type*, Defs) const override;

    u64 dim_;

    friend class World;
};

bool use_lea(const Type*);

//------------------------------------------------------------------------------

class TypeTable {
public:
    explicit TypeTable(World& world);

private:
    World& world_;

    const Type* star_;
    const TupleType* unit_; ///< tuple().
    const FnType* fn0_;
    const BottomType* bottom_ty_;
    const MemType* mem_;
    const FrameType* frame_;
    const PrimType* primtypes_[Num_PrimTypes];

    friend class World;
};

//------------------------------------------------------------------------------

inline bool is_mem        (const Def* def) { return def->type()->isa<MemType>(); }

//------------------------------------------------------------------------------

}

#endif
