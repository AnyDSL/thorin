#ifndef ANYDSL2_TYPE_H
#define ANYDSL2_TYPE_H

#include <boost/functional/hash.hpp>
#include <boost/tuple/tuple.hpp>

#include "anydsl2/node.h"
#include "anydsl2/util/array.h"

#define ANYDSL2_TYPE_HASH_EQUAL \
    virtual bool equal(const Node* other) const { return equal_type(tuple(), other->as<Type>()); } \
    virtual size_t hash() const { return hash_type(tuple()); }

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

inline std::ostream& operator << (std::ostream& o, const GenericMap& map) { 
    o << map.to_string(); return o; 
}

//------------------------------------------------------------------------------

typedef boost::tuple<int> TypeTuple0;
typedef boost::tuple<int, const Type*> TypeTuple1;
typedef boost::tuple<int, ArrayRef<const Type*> > TypeTupleN;

size_t hash_type(const TypeTuple0& tuple);
size_t hash_type(const TypeTuple1& tuple);
size_t hash_type(const TypeTupleN& tuple);

bool equal_type(const TypeTuple0&, const Type*);
bool equal_type(const TypeTuple1&, const Type*);
bool equal_type(const TypeTupleN&, const Type*);

class Type : public Node {
protected:

    Type(World& world, int kind, size_t num, bool is_generic)
        : Node(kind, num, "")
        , world_(world)
        , is_generic_(is_generic)
    {}

public:

    void dump() const;
    void dump(bool fancy) const;
    World& world() const { return world_; }
    Elems elems() const { return ops_ref<const Type*>(); }
    const Type* elem(size_t i) const { return elems()[i]; }
    const Type* elem_via_lit(const Def* def) const;
    const Ptr* to_ptr() const;
    virtual void vdump(Printer &printer) const = 0;
    bool check_with(const Type* type) const;
    bool infer_with(GenericMap& map, const Type* type) const;
    const Type* specialize(const GenericMap& generic_map) const;
    bool is_generic() const { return is_generic_; }
    int order() const;
    TypeTuple0 tuple() const { return TypeTuple0(kind()); }

private:

    World& world_;

protected:

    ANYDSL2_TYPE_HASH_EQUAL
    bool is_generic_;

    friend class Def;
    friend class TypeHash;
    friend class TypeEqual;
};

std::ostream& operator << (std::ostream& o, const anydsl2::Type* type);

bool is_generic(ArrayRef<const Type*> elems);

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
    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

/// The type of a stack frame.
class Frame : public Type {
private:

    Frame(World& world)
        : Type(world, Node_Frame, 0, false)
    {}
    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind kind);

public:

    PrimTypeKind primtype_kind() const { return (PrimTypeKind) node_kind(); }

    bool is_int()   const { return anydsl2::is_int(primtype_kind()); }
    bool is_float() const { return anydsl2::is_float(primtype_kind()); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Ptr : public Type {
private:

    Ptr(const Type* ref)
        : Type(ref->world(), Node_Ptr, 1, ref->is_generic())
    {
        set(0, ref);
    }

    virtual void vdump(Printer& printer) const;
    ANYDSL2_TYPE_HASH_EQUAL

public:

    const Type* ref() const { return elem(0); }
    TypeTuple1 tulpe() const { return TypeTuple1(kind(), ref()); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    CompoundType(World& world, int kind, size_t num_elems);
    CompoundType(World& world, int kind, ArrayRef<const Type*> elems);

    void dump_inner(Printer& printer) const;
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

public:

    bool named() const { return named_; }
    // TODO build setter for named sigmas which sets is_generic_

private:

    virtual void vdump(Printer& printer) const;
    virtual size_t hash() const;
    virtual bool equal(const Node* other) const;

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
    TypeTupleN tuple() const { return TypeTupleN(kind(), elems()); }
    ANYDSL2_TYPE_HASH_EQUAL

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class IndexType : public Type {
protected:

    IndexType(World& world, int kind, size_t index, bool is_generic)
        : Type(world, kind, 0, is_generic)
        , index_(index)
    {}

public:

    size_t index() const { return index_; }

private:

    virtual size_t hash() const;
    virtual bool equal(const Node* other) const;

    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

typedef boost::tuple<int, size_t> GenericTuple;
size_t hash_type(const GenericTuple& tuple);
bool equal_type(const GenericTuple&, const Type*);

class Generic : public IndexType {
private:

    Generic(World& world, size_t index)
        : IndexType(world, Node_Generic, index, true)
    {}
    ANYDSL2_TYPE_HASH_EQUAL

public:

    GenericTuple tuple() const { return GenericTuple(kind(), index()); }
    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Opaque : public IndexType {
private:

    Opaque(World& world, size_t index)
        : IndexType(world, Node_Opaque, index, true)
    {}

public:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

inline bool is_u1(const Type* type) { 
    if (const PrimType* p = type->isa<PrimType>())
        return p->primtype_kind() == PrimType_u1;
    return false;
}

inline bool is_int(const Type* type) { 
    if (const PrimType* p = type->isa<PrimType>())
        return is_int(p->primtype_kind());
    return false;
}

inline bool is_float(const Type* type) { 
    if (const PrimType* p = type->isa<PrimType>())
        return is_float(p->primtype_kind());
    return false;
}

inline bool is_primtype(const Type* type) { return is_primtype(type->kind()); }

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
