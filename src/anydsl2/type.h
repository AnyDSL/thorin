#ifndef ANYDSL2_TYPE_H
#define ANYDSL2_TYPE_H

#include "anydsl2/node.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/hash.h"

namespace anydsl2 {

class Def;
class Generic;
class Lambda;
class Pi;
class PrimLit;
class Ptr;
class Type;
class World;

//------------------------------------------------------------------------------

class GenericMap {
public:

    GenericMap() {}

    const Type*& operator [] (const Generic* generic) const;
    bool is_empty() const;
    const char* to_string() const;

private:

    mutable std::vector<const Type*> types_;
};

inline std::ostream& operator << (std::ostream& o, const GenericMap& map) { o << map.to_string(); return o; }

//------------------------------------------------------------------------------

class Type : public Node {
protected:

    Type(World& world, int kind, size_t num, bool is_generic)
        : Node(kind, num, "")
        , world_(world)
        , is_generic_(is_generic)
    {}

public:

    void dump() const;
    World& world() const { return world_; }
    ArrayRef<const Type*> elems() const { return ops_ref<const Type*>(); }
    const Type* elem(size_t i) const { return elems()[i]; }
    const Type* elem_via_lit(const Def* def) const;
    bool check_with(const Type* type) const;
    bool infer_with(GenericMap& map, const Type* type) const;
    const Type* specialize(const GenericMap& generic_map) const;
    bool is_generic() const { return is_generic_; }
    bool is_u1() const { return kind() == PrimType_u1; }
    bool is_int() const { return anydsl2::is_int(kind()); }
    bool is_float() const { return anydsl2::is_float(kind()); }
    bool is_primtype() const { return anydsl2::is_primtype(kind()); }
    int order() const;
    /**
     * Returns the vector length.
     * Raises an assertion if type of this is not a \p VectorType.
     */
    size_t length() const;

private:

    World& world_;

protected:

    bool is_generic_;

    friend class Def;
    friend struct TypeHash;
    friend struct TypeEqual;
};

struct TypeHash : std::unary_function<const Type*, size_t> {
    size_t operator () (const Type* t) const { return t->hash(); }
};

struct TypeEqual : std::binary_function<const Type*, const Type*, bool> {
    bool operator () (const Type* t1, const Type* t2) const { return t1->equal(t2); }
};

//------------------------------------------------------------------------------

/// The type of the memory monad.
class Mem : public Type {
private:

    Mem(World& world)
        : Type(world, Node_Mem, 0, false)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

/// The type of a stack frame.
class Frame : public Type {
private:

    Frame(World& world)
        : Type(world, Node_Frame, 0, false)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

class VectorType : public Type {
protected:

    VectorType(World& world, int kind, size_t num_elems, size_t length, bool is_generic)
        : Type(world, kind, num_elems, is_generic)
        , length_(length)
    {}

    virtual size_t hash() const { return hash_combine(Type::hash(), length()); }
    virtual bool equal(const Node* other) const { 
        return Type::equal(other) ? this->length() == other->as<VectorType>()->length() : false;
    }

public:

    /// The number of vector elements - the vector length.
    size_t length() const { return length_; }
    bool is_vector() const { return length_ != 1; }

private:

    size_t length_;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public VectorType {
private:

    PrimType(World& world, PrimTypeKind kind, size_t length)
        : VectorType(world, (int) kind, 0, length, false)
    {}

public:

    PrimTypeKind primtype_kind() const { return (PrimTypeKind) node_kind(); }

private:

    friend class World;
};

//------------------------------------------------------------------------------

class Ptr : public VectorType {
private:

    Ptr(World& world, const Type* referenced_type, size_t length)
        : VectorType(world, (int) Node_Ptr, 1, length, referenced_type->is_generic())
    {
        set(0, referenced_type);
    }

public:

    const Type* referenced_type() const { return elem(0); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    CompoundType(World& world, int kind, size_t num_elems);
    CompoundType(World& world, int kind, ArrayRef<const Type*> elems);
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, size_t size, const std::string& sigma_name)
        : CompoundType(world, Node_Sigma, size)
        , named_(true)
    {
        name = sigma_name;
    }
    Sigma(World& world, ArrayRef<const Type*> elems)
        : CompoundType(world, Node_Sigma, elems)
        , named_(false)
    {}

    virtual size_t hash() const { return named_ ? hash_value(this) : CompoundType::hash(); }
    virtual bool equal(const Node* other) const { return named_ ? this == other : CompoundType::equal(other); }

public:

    bool named() const { return named_; }
    // TODO build setter for named sigmas which sets is_generic_

private:

    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public CompoundType {
private:

    Pi(World& world, ArrayRef<const Type*> elems)
        : CompoundType(world, Node_Pi, elems)
    {}

public:

    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    friend class World;
};

//------------------------------------------------------------------------------

class Generic : public Type {
private:

    Generic(World& world, size_t index)
        : Type(world, Node_Generic, 0, true)
        , index_(index)
    {}

    virtual size_t hash() const { return hash_combine(Type::hash(), index()); }
    virtual bool equal(const Node* other) const { 
        return Type::equal(other) ? index() == other->as<Generic>()->index() : false; 
    }

public:

    size_t index() const { return index_; }

private:

    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

class GenericRef : public Type {
private:

    GenericRef(World& world, const Generic* generic, Lambda* lambda);
    virtual ~GenericRef();

    virtual size_t hash() const { return hash_combine(Type::hash(), lambda()); }
    virtual bool equal(const Node* other) const { 
        return Type::equal(other) ? lambda() == other->as<GenericRef>()->lambda() : false; 
    }

public:

    const Generic* generic() const { return elem(0)->as<Generic>(); }
    Lambda* lambda() const { return lambda_; }

private:

    Lambda* lambda_;

    friend class World;
};

//------------------------------------------------------------------------------

class GenericBuilder {
public:

    GenericBuilder(World& world)
        : world_(world)
        , index_(0)
    {}

    size_t new_def();
    const Generic* use(size_t handle);
    void pop();

private:

    World& world_;
    size_t index_;
    typedef std::vector<const Generic*> Index2Generic;
    Index2Generic index2generic_;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
