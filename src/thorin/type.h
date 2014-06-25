#ifndef THORIN_TYPE_H
#define THORIN_TYPE_H

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"

namespace thorin {

class Def;
class World;

//------------------------------------------------------------------------------

template<class T>
class Proxy {
public:
    typedef T BaseType;

    Proxy()
        : node_(nullptr)
    {}
    Proxy(const T* node)
        : node_(node)
    {}
    Proxy(Proxy<T>&& other)
        : node_(std::move(other.node_))
    {
        other.node_ = nullptr;
    }
    Proxy(const Proxy<T>& other)
        : node_(other.node_)
    {}

    bool empty() const { return node_ == nullptr; }
    bool operator == (const Proxy<T>& other) const {
        assert(&node()->world() == &other.node()->world());
        return this->node()->unify() == other.node()->unify();
    }
    bool operator != (const Proxy<T>& other) const { return !(*this == other); }
    const T* representative() const { return node()->representative()->template as<T>(); }
    const T* node() const { assert(node_ != nullptr); return node_; }
    const T* operator  * () const { return node()->is_unified() ? representative() : node(); }
    const T* operator -> () const { return *(*this); }
    /// Automatic up-cast in the class hierarchy.
    template<class U> operator Proxy<U>() {
        static_assert(std::is_base_of<U, T>::value, "U is not a base type of T");
        return Proxy<U>((**this)->template as<T>());
    }
    template<class U> Proxy<typename U::BaseType> isa() const {
        return Proxy<typename U::BaseType>((*this)->template isa<typename U::BaseType>());
    }
    template<class U> Proxy<typename U::BaseType> as() const {
        return Proxy<typename U::BaseType>((*this)->template as <typename U::BaseType>());
    }
    operator bool() { return !empty(); }
    void clear() { assert(node_ != nullptr); node_ = nullptr; }
    Proxy<T> unify() const { return node()->unify()->template as<T>(); }
    Proxy<T>& operator= (Proxy<T> other) { swap(*this, other); return *this; }
    friend void swap(Proxy<T>& p1, Proxy<T>& p2) {
        assert(p1.node_ == nullptr);
        auto tmp = p2.node();
        p2.node_ = p1.node_;
        p1.node_ = tmp;
    }

private:
    const T* node_;
};

class TypeNode;                 typedef Proxy<TypeNode>                 Type;
class MemTypeNode;              typedef Proxy<MemTypeNode>              MemType;
class FrameTypeNode;            typedef Proxy<FrameTypeNode>            FrameType;
class VectorTypeNode;           typedef Proxy<VectorTypeNode>           VectorType;
class PrimTypeNode;             typedef Proxy<PrimTypeNode>             PrimType;
class PtrTypeNode;              typedef Proxy<PtrTypeNode>              PtrType;
class TupleTypeNode;            typedef Proxy<TupleTypeNode>            TupleType;
class StructAbsTypeNode;        typedef Proxy<StructAbsTypeNode>        StructAbsType;
class StructAppTypeNode;        typedef Proxy<StructAppTypeNode>        StructAppType;
class FnTypeNode;               typedef Proxy<FnTypeNode>               FnType;
class ArrayTypeNode;            typedef Proxy<ArrayTypeNode>            ArrayType;
class DefiniteArrayTypeNode;    typedef Proxy<DefiniteArrayTypeNode>    DefiniteArrayType;
class IndefiniteArrayTypeNode;  typedef Proxy<IndefiniteArrayTypeNode>  IndefiniteArrayType;
class TypeVarNode;              typedef Proxy<TypeVarNode>              TypeVar;

template<class T> struct GIDHash;
template<class T> struct GIDEq;
template<class To>
using TypeMap    = HashMap<const TypeNode*, To, GIDHash<const TypeNode*>, GIDEq<const TypeNode*>>;
using TypeSet    = HashSet<const TypeNode*, GIDHash<const TypeNode*>, GIDEq<const TypeNode*>>;
using Type2Type  = TypeMap<const TypeNode*>;
using TypeVarSet = HashSet<const TypeVarNode*, GIDHash<const TypeVarNode*>, GIDEq<const TypeVarNode*>>;

//------------------------------------------------------------------------------

class TypeNode : public MagicCast<TypeNode> {
private:
    TypeNode& operator = (const TypeNode&); ///< Do not copy-assign a \p TypeNode instance.
    TypeNode(const TypeNode&);              ///< Do not copy-construct a \p TypeNode.

protected:
    TypeNode(World& world, NodeKind kind, ArrayRef<Type> args)
        : representative_(nullptr)
        , world_(world)
        , kind_(kind)
        , args_(args.size())
        , gid_(-1)
    {
        for (size_t i = 0, e = num_args(); i != e; ++i) {
            if (auto arg = args[i])
                set(i, arg);
        }
    }

    void set(size_t i, Type type) { args_[i] = type; }

public:
    NodeKind kind() const { return kind_; }
    bool is_corenode() const { return ::thorin::is_corenode(kind()); }
    ArrayRef<Type> args() const { return args_; }
    ArrayRef<TypeVar> type_vars() const { return type_vars_; }
    size_t num_type_vars() const { return type_vars().size(); }
    Type arg(size_t i) const { assert(i < args().size()); return args()[i]; }
    TypeVar type_var(size_t i) const { assert(i < type_vars().size()); return type_vars()[i]; }
    void bind(TypeVar v) const;
    size_t num_args() const { return args_.size(); }
    bool is_polymorphic() const { return num_type_vars() > 0; }
    Type arg_via_lit(const Def& def) const;
    bool empty() const { return args_.empty(); }
    void dump() const;
    World& world() const { return world_; }
    bool check_with(Type) const { return true; } // TODO
    bool infer_with(Type2Type&, Type) const { return true; } // TODO
    const TypeNode* representative() const { return representative_; }
    bool is_unified() const { return representative_ != nullptr; }
    const TypeNode* unify() const;
    void free_type_vars(TypeVarSet& bound, TypeVarSet& free) const;
    TypeVarSet free_type_vars() const;
    size_t gid() const { return gid_; }
    int order() const;
    /// Returns the vector length. Raises an assertion if this type is not a \p VectorType.
    size_t length() const;
    Type instantiate(ArrayRef<Type>) const;
    Type instantiate(Type2Type&) const;
    Type specialize(Type2Type&) const;

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

    virtual size_t hash() const;
    virtual bool equal(const TypeNode*) const;
    virtual bool is_closed() const;
    virtual IndefiniteArrayType is_indefinite() const;

protected:
    Array<Type> specialize_args(Type2Type&) const;

private:
    virtual Type vinstantiate(Type2Type&) const = 0;

    mutable const TypeNode* representative_;
    World& world_;
    NodeKind kind_;
    mutable std::vector<TypeVar> type_vars_;
    std::vector<Type> args_;
    mutable size_t gid_;

    friend class World;
};

/// The type of the memory monad.
class MemTypeNode : public TypeNode {
private:
    MemTypeNode(World& world)
        : TypeNode(world, Node_MemType, {})
    {}

    virtual Type vinstantiate(Type2Type&) const override;

    friend class World;
};

/// The type of a stack frame.
class FrameTypeNode : public TypeNode {
private:
    FrameTypeNode(World& world)
        : TypeNode(world, Node_FrameType, {})
    {}

    virtual Type vinstantiate(Type2Type&) const override;

    friend class World;
};

class VectorTypeNode : public TypeNode {
protected:
    VectorTypeNode(World& world, NodeKind kind, ArrayRef<Type> args, size_t length)
        : TypeNode(world, kind, args)
        , length_(length)
    {}

    virtual size_t hash() const override { return hash_combine(TypeNode::hash(), length()); }
    virtual bool equal(const TypeNode* other) const override {
        return TypeNode::equal(other) && this->length() == other->as<VectorTypeNode>()->length();
    }

public:
    /// The number of vector argents - the vector length.
    size_t length() const { return length_; }
    bool is_vector() const { return length_ != 1; }
    /// Rebuilds the type with vector length 1.
    VectorType scalarize() const;

private:
    size_t length_;
};

/// Primitive types -- also known as atomic or scalar types.
class PrimTypeNode : public VectorTypeNode {
private:
    PrimTypeNode(World& world, PrimTypeKind kind, size_t length)
        : VectorTypeNode(world, (NodeKind) kind, {}, length)
    {}

public:
    PrimTypeKind primtype_kind() const { return (PrimTypeKind) kind(); }

private:
    virtual Type vinstantiate(Type2Type&) const override;

    friend class World;
};

enum class AddressSpace : uint32_t {
    Generic  = 0,
    Global   = 1,
    Texture  = 2,
    Shared   = 3,
    Constant = 4,
};

class PtrTypeNode : public VectorTypeNode {
private:
    PtrTypeNode(World& world, Type referenced_type, size_t length, uint32_t device, AddressSpace addr_space)
        : VectorTypeNode(world, Node_PtrType, {referenced_type}, length)
        , addr_space_(addr_space)
        , device_(device)
    {}

public:
    Type referenced_type() const { return arg(0); }
    AddressSpace addr_space() const { return addr_space_; }
    uint32_t device() const { return device_; }
    bool is_host_device() const { return device_ == 0; }

    virtual size_t hash() const override;
    virtual bool equal(const TypeNode* other) const override;

private:
    virtual Type vinstantiate(Type2Type&) const override;

    AddressSpace addr_space_;
    uint32_t device_;

    friend class World;
};

class StructAbsTypeNode : public TypeNode {
private:
    StructAbsTypeNode(World& world, size_t size, const std::string& name)
        : TypeNode(world, Node_StructAbsType, Array<Type>(size))
        , name_(name)
    {}

public:
    const std::string& name() const { return name_; }
    void set(size_t i, Type type) const { const_cast<StructAbsTypeNode*>(this)->TypeNode::set(i, type); }
    virtual size_t hash() const override { return hash_value(this->gid()); }
    virtual bool equal(const TypeNode* other) const override { return this == other; }

private:
    virtual Type vinstantiate(Type2Type&) const override;

    std::string name_;

    friend class World;
};

class StructAppTypeNode : public TypeNode {
private:
    StructAppTypeNode(StructAbsType struct_abs_type, ArrayRef<Type> args)
        : TypeNode(struct_abs_type->world(), Node_StructAppType, args)
        , struct_abs_type_(struct_abs_type)
    {}

    virtual size_t hash() const override;
    virtual bool equal(const TypeNode* other) const override;

public:
    StructAbsType struct_abs_type() const { return struct_abs_type_; }

private:
    virtual Type vinstantiate(Type2Type&) const override;

    StructAbsType struct_abs_type_;
    mutable Array<Type> elem_cache_;

    friend class World;
};

class TupleTypeNode : public TypeNode {
private:
    TupleTypeNode(World& world, ArrayRef<Type> args)
        : TypeNode(world, Node_TupleType, args)
    {}

    virtual Type vinstantiate(Type2Type&) const override;

    friend class World;
};

class FnTypeNode : public TypeNode {
private:
    FnTypeNode(World& world, ArrayRef<Type> args)
        : TypeNode(world, Node_FnType, args)
    {}

public:
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

private:
    virtual Type vinstantiate(Type2Type&) const override;

    friend class World;
};

class ArrayTypeNode : public TypeNode {
protected:
    ArrayTypeNode(World& world, NodeKind kind, Type elem_type)
        : TypeNode(world, kind, {elem_type})
    {}

public:
    Type elem_type() const { return arg(0); }
};

class IndefiniteArrayTypeNode : public ArrayTypeNode {
public:
    IndefiniteArrayTypeNode(World& world, Type elem_type)
        : ArrayTypeNode(world, Node_IndefiniteArrayType, elem_type)
    {}

    virtual IndefiniteArrayType is_indefinite() const;

private:
    virtual Type vinstantiate(Type2Type&) const override;

    friend class World;
};

class DefiniteArrayTypeNode : public ArrayTypeNode {
public:
    DefiniteArrayTypeNode(World& world, Type elem_type, u64 dim)
        : ArrayTypeNode(world, Node_DefiniteArrayType, elem_type)
        , dim_(dim)
    {}

    u64 dim() const { return dim_; }
    virtual size_t hash() const override { return hash_combine(TypeNode::hash(), dim()); }
    virtual bool equal(const TypeNode* other) const override {
        return TypeNode::equal(other) && this->dim() == other->as<DefiniteArrayTypeNode>()->dim();
    }

private:
    virtual Type vinstantiate(Type2Type&) const override;

    u64 dim_;

    friend class World;
};

class TypeVarNode : public TypeNode {
private:
    TypeVarNode(World& world)
        : TypeNode(world, Node_TypeVar, {})
        , equiv_(nullptr)
    {}

public:
    Type bound_at() const { return Type(bound_at_); }
    virtual bool equal(const TypeNode*) const override;
    virtual bool is_closed() const override { return bound_at_ != nullptr; }

private:
    virtual Type vinstantiate(Type2Type&) const override;

    mutable const TypeNode* bound_at_;
    mutable const TypeVarNode* equiv_;

    friend bool TypeNode::equal(const TypeNode*) const;
    friend void TypeNode::bind(TypeVar type_var) const;
    friend class World;
};

//------------------------------------------------------------------------------

}

#endif
