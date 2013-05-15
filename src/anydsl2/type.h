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
class Printer;
class Ptr;
class Type;
class World;

typedef ArrayRef<const Type*> Elems;

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
    Elems elems() const { return ops_ref<const Type*>(); }
    const Type* elem(size_t i) const { return elems()[i]; }
    const Type* elem_via_lit(const Def* def) const;
    const Ptr* to_ptr() const;
    bool check_with(const Type* type) const;
    bool infer_with(GenericMap& map, const Type* type) const;
    const Type* specialize(const GenericMap& generic_map) const;
    bool is_generic() const { return is_generic_; }
    int order() const;

    bool is_u1() const { return kind() == PrimType_u1; }
    bool is_int() const { return anydsl2::is_int(kind()); }
    bool is_float() const { return anydsl2::is_float(kind()); }
    bool is_primtype() const { return anydsl2::is_primtype(kind()); }

private:

    World& world_;

protected:

    bool is_generic_;

    friend class Def;
    friend struct TypeHash;
    friend struct TypeEqual;
};

std::ostream& operator << (std::ostream& o, const anydsl2::Type* type);

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

    virtual Printer& print(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

/// The type of a stack frame.
class Frame : public Type {
private:

    Frame(World& world)
        : Type(world, Node_Frame, 0, false)
    {}

    virtual Printer& print(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind kind, size_t num_elems)
        : Type(world, (int) kind, 0, false)
        , num_elems_(num_elems)
    {
        assert(num_elems == 1);
    }

    virtual Printer& print(Printer& printer) const;
    virtual size_t hash() const { return hash_combine(Type::hash(), num_elems()); }
    virtual bool equal(const Node* other) const { 
        return Type::equal(other) ? this->num_elems() == other->as<PrimType>()->num_elems() : false;
    }

public:

    size_t num_elems() const { return num_elems_; }
    bool is_vector() const { return num_elems_ != 1; }
    PrimTypeKind primtype_kind() const { return (PrimTypeKind) node_kind(); }

private:

    size_t num_elems_;

    friend class World;
};

//------------------------------------------------------------------------------

class Ptr : public Type {
private:

    Ptr(World& world, const Type* referenced_type, size_t num_elems)
        : Type(world, Node_Ptr, 1, referenced_type->is_generic())
        , num_elems_(num_elems)
    {
        set(0, referenced_type);
        assert(num_elems == 1);
    }

    virtual Printer& print(Printer& printer) const;
    virtual size_t hash() const { return hash_combine(Type::hash(), num_elems()); }
    virtual bool equal(const Node* other) const { 
        return Type::equal(other) ? this->num_elems() == other->as<Ptr>()->num_elems() : false;
    }

public:

    const Type* referenced_type() const { return elem(0); }
    size_t num_elems() const { return num_elems_; }
    bool is_vector() const { return num_elems_ != 1; }

private:

    size_t num_elems_;

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    CompoundType(World& world, int kind, size_t num_subtypes);
    CompoundType(World& world, int kind, Elems elems);
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
    Sigma(World& world, Elems elems)
        : CompoundType(world, Node_Sigma, elems)
        , named_(false)
    {}

    virtual Printer& print(Printer& printer) const;
    virtual size_t hash() const;
    virtual bool equal(const Node* other) const;

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

    Pi(World& world, Elems elems)
        : CompoundType(world, Node_Pi, elems)
    {}

    virtual Printer& print(Printer& printer) const;

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

    virtual Printer& print(Printer& printer) const;
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

class Opaque : public CompoundType {
private:

    Opaque(World& world, Elems elems, ArrayRef<uint32_t> flags)
        : CompoundType(world, Node_Opaque, elems.size())
        , flags_(flags)
    {}

    virtual Printer& print(Printer& printer) const;
    virtual size_t hash() const;
    virtual bool equal(const Node* other) const {
        return Type::equal(other) ? flags() == other->as<Opaque>()->flags() : false;
    }

public:

    ArrayRef<uint32_t> flags() const { return flags_; }
    uint32_t flag(size_t i) const { return flags_[i]; }
    size_t num_flags() const { return flags_.size(); }

private:

    Array<uint32_t> flags_;

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
