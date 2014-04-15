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
    explicit Proxy(const T* node)
        : node_(node)
    {}

    bool empty() const { return node_ == nullptr; }
    bool operator == (const Proxy<T>& other) const {
#if 0
        assert(node_ != nullptr);         
        assert(&node()->typetable() == &other.node()->typetable());
        unify(node()->typetable(), *this);
        unify(node()->typetable(), other);
#endif
        return representative() == other.representative();
    }
    bool operator != (const Proxy<T>& other) const { assert(node_ != nullptr); return !(*this == other); }
    T* representative() const { assert(node_ != nullptr); return node_->representative()->template as<T>(); }
    T* node() const { assert(node_ != nullptr); return node_; }
    //T* operator  * () const { assert(node_ != nullptr); return node_->is_unified() ? representative() : node_->template as<T>(); }
    T* operator  * () const { assert(node_ != nullptr); return representative(); }
    T* operator -> () const { assert(node_ != nullptr); return *(*this); }
    /// Automatic up-cast in the class hierarchy.
    template<class U> operator Proxy<U>() {
        static_assert(std::is_base_of<U, T>::value, "U is not a base type of T");
        assert(node_ != nullptr); return Proxy<U>((U*) node_);
    }
    template<class U> Proxy<typename U::BaseType> isa() const { 
        assert(node_ != nullptr); return Proxy<typename U::BaseType>((*this)->isa<typename U::BaseType>()); 
    }
    template<class U> Proxy<typename U::BaseType> as() const { 
        assert(node_ != nullptr); return Proxy<typename U::BaseType>((*this)->as <typename U::BaseType>()); 
    }
    operator bool() { return !empty(); }
    Proxy<T>& operator= (Proxy<T> other) { 
        assert(node_ == nullptr);
        node_ = *other; 
        return *this; 
    }
    void clear() { node_ = nullptr; }

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
class StructTypeNode;           typedef Proxy<StructTypeNode>           StructType;
class FnTypeNode;               typedef Proxy<FnTypeNode>               FnType;
class ArrayTypeNode;            typedef Proxy<ArrayTypeNode>            ArrayType;
class DefiniteArrayTypeNode;    typedef Proxy<DefiniteArrayTypeNode>    DefiniteArrayType;
class IndefiniteArrayTypeNode;  typedef Proxy<IndefiniteArrayTypeNode>  IndefiniteArrayType;
class TypeNodeVar;              typedef Proxy<TypeNodeVar>              TypeVar;

template<class T> struct GIDHash;
template<class T> struct GIDEq;
template<class To> 
using TypeMap   = HashMap<const TypeNode*, To, GIDHash<const TypeNode*>, GIDEq<const TypeNode*>>;
using Type2Type = TypeMap<const TypeNode*>;

class TypeNode : public MagicCast<TypeNode> {
private:
    TypeNode& operator = (const TypeNode&); ///< Do not copy-assign a \p TypeNode instance.
    TypeNode(const TypeNode&);              ///< Do not copy-construct a \p TypeNode.

protected:
    TypeNode(World& world, NodeKind kind, size_t num, bool is_generic)
        : representative_(this)
        , world_(world)
        , kind_(kind)
        , elems_(num)
        , gid_(-1)
        , is_generic_(is_generic)
    {}

    void set(size_t i, Type type) { elems_[i] = type; }

public:
    NodeKind kind() const { return kind_; }
    bool is_corenode() const { return ::thorin::is_corenode(kind()); }
    ArrayRef<Type> elems() const { return elems_; }
    Type elem(size_t i) const { assert(i < elems().size()); return elems()[i]; }
    Type elem_via_lit(const Def& def) const;
    size_t size() const { return elems_.size(); }
    bool empty() const { return elems_.empty(); }
    void dump() const;
    World& world() const { return world_; }
    bool check_with(Type) const { return true; } // TODO
    bool infer_with(Type2Type&, Type) const { return true; } // TODO
    Type specialize(const Type2Type&) const { return Type(this); } // TODO
    bool is_generic() const { return is_generic_; }
    TypeNode* representative() const { return representative_; }

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

    size_t gid() const { return gid_; }
    int order() const;
    virtual size_t hash() const;
    virtual bool equal(const TypeNode*) const;
    /**
     * Returns the vector length.
     * Raises an assertion if type of this is not a \p VectorType.
     */
    size_t length() const;

private:
    TypeNode* representative_;
    World& world_;
    NodeKind kind_;
    std::vector<Type> elems_;
    mutable size_t gid_;

protected:
    bool is_generic_;

    friend class World;
};

struct TypeHash { size_t operator () (const TypeNode* t) const { return t->hash(); } };
struct TypeEqual { bool operator () (const TypeNode* t1, const TypeNode* t2) const { return t1->equal(t2); } };

//------------------------------------------------------------------------------

/// The type of the memory monad.
class MemTypeNode : public TypeNode {
private:
    MemTypeNode(World& world)
        : TypeNode(world, Node_MemType, 0, false)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

/// The type of a stack frame.
class FrameTypeNode : public TypeNode {
private:
    FrameTypeNode(World& world)
        : TypeNode(world, Node_FrameType, 0, false)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

class VectorTypeNode : public TypeNode {
protected:
    VectorTypeNode(World& world, NodeKind kind, size_t num_elems, size_t length, bool is_generic)
        : TypeNode(world, kind, num_elems, is_generic)
        , length_(length)
    {}

    virtual size_t hash() const override { return hash_combine(TypeNode::hash(), length()); }
    virtual bool equal(const TypeNode* other) const override { 
        return TypeNode::equal(other) && this->length() == other->as<VectorTypeNode>()->length();
    }

public:
    /// The number of vector elements - the vector length.
    size_t length() const { return length_; }
    bool is_vector() const { return length_ != 1; }
    /// Rebuilds the type with vector length 1.
    VectorType scalarize() const;

private:
    size_t length_;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimTypeNode : public VectorTypeNode {
private:
    PrimTypeNode(World& world, PrimTypeKind kind, size_t length)
        : VectorTypeNode(world, (NodeKind) kind, 0, length, false)
    {}

public:
    PrimTypeKind primtype_kind() const { return (PrimTypeKind) kind(); }

private:
    friend class World;
};

//------------------------------------------------------------------------------

enum class AddressSpace : uint32_t {
    Global  = 0,
    Texture = 1,
    Shared  = 2,
};

class PtrTypeNode : public VectorTypeNode {
private:
    PtrTypeNode(World& world, Type referenced_type, size_t length, uint32_t device, AddressSpace addr_space)
        : VectorTypeNode(world, Node_PtrType, 1, length, referenced_type->is_generic())
        , addr_space_(addr_space)
        , device_(device)
    {
        set(0, referenced_type);
    }

public:
    Type referenced_type() const { return elem(0); }
    AddressSpace addr_space() const { return addr_space_; }
    uint32_t device() const { return device_; }
    bool is_host_device() const { return device_ == 0; }

    virtual size_t hash() const override;
    virtual bool equal(const TypeNode* other) const override;

private:
    AddressSpace addr_space_;
    uint32_t device_;

    friend class World;
};

class StructTypeNode : public TypeNode {
private:
    StructTypeNode(World& world, size_t size, const std::string& name)
        : TypeNode(world, Node_StructType, size, false)
        , name_(name)
    {}

    virtual size_t hash() const override { return hash_value(this->gid()); }
    virtual bool equal(const TypeNode* other) const override { return this == other; }

public:
    const std::string& name() const { return name_; }
    void set(size_t i, Type type) { TypeNode::set(i, type); is_generic_ |= type->is_generic(); }

private:
    std::string name_;

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundTypeNode : public TypeNode {
protected:
    CompoundTypeNode(World& world, NodeKind kind, ArrayRef<Type> elems);
};

class TupleTypeNode : public CompoundTypeNode {
private:
    TupleTypeNode(World& world, ArrayRef<Type> elems)
        : CompoundTypeNode(world, Node_TupleType, elems)
    {}

    friend class World;
};

/// A function type.
class FnTypeNode : public CompoundTypeNode {
private:
    FnTypeNode(World& world, ArrayRef<Type> elems)
        : CompoundTypeNode(world, Node_FnType, elems)
    {}

public:
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    friend class World;
};

//------------------------------------------------------------------------------

class ArrayTypeNode : public TypeNode {
protected:
    ArrayTypeNode(World& world, NodeKind kind, Type elem_type)
        : TypeNode(world, kind, 1, elem_type->is_generic())
    {
        set(0, elem_type);
    }

public:
    Type elem_type() const { return elem(0); }
};

class IndefiniteArrayTypeNode : public ArrayTypeNode {
public:
    IndefiniteArrayTypeNode(World& world, Type elem_type)
        : ArrayTypeNode(world, Node_IndefiniteArrayType, elem_type)
    {}

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
    u64 dim_;

    friend class World;
};

//------------------------------------------------------------------------------

class TypeNodeVar : public TypeNode {
private:
    TypeNodeVar(World& world, size_t index)
        : TypeNode(world, Node_TypeVar, 0, true)
        , index_(index)
    {}

    virtual size_t hash() const { return hash_combine(TypeNode::hash(), index()); }
    virtual bool equal(const TypeNode* other) const { 
        return TypeNode::equal(other) && index() == other->as<TypeNodeVar>()->index();
    }

public:
    size_t index() const { return index_; }

private:
    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

}

#endif
