#ifndef ANYDSL2_TYPE_H
#define ANYDSL2_TYPE_H

#include <boost/tuple/tuple_comparison.hpp>

#include "anydsl2/node.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/hash.h"

#define ANYDSL2_TYPE_HASH_EQUAL \
    virtual bool equal(const Type* other) const { \
        typedef BOOST_TYPEOF(*this) T; \
        return other->isa<T>() && this->as_tuple() == other->as<T>()->as_tuple(); \
    } \
    virtual size_t hash() const { return hash_tuple(as_tuple()); }

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

class Type : public Node {
protected:

    Type(World& world, int kind, size_t num, bool is_generic)
        : Node(kind, num, "")
        , world_(world)
        , is_generic_(is_generic)
    {}

public:

    void dump(bool fancy) const;
    void dump() const;
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
    TypeTuple0 as_tuple() const { return TypeTuple0(kind()); }

    bool is_u1() const { return kind() == PrimType_u1; }
    bool is_int() const { return anydsl2::is_int(kind()); }
    bool is_float() const { return anydsl2::is_float(kind()); }
    bool is_primtype() const { return anydsl2::is_primtype(kind()); }

//------------------------------------------------------------------------------

private:

    World& world_;

protected:

    ANYDSL2_TYPE_HASH_EQUAL
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

    PrimType(World& world, const TypeTuple0& args)
        : Type(world, args.get<0>(), 0, false)
    {}

public:

    PrimTypeKind primtype_kind() const { return (PrimTypeKind) node_kind(); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

class Ptr : public Type {
private:

    Ptr(World& world, const TypeTuple1& args)
        : Type(world, args.get<0>(), 1, args.get<1>()->is_generic())
    {
        set(0, args.get<1>());
    }

    virtual void vdump(Printer& printer) const;
    ANYDSL2_TYPE_HASH_EQUAL

public:

    const Type* ref() const { return elem(0); }
    TypeTuple1 as_tuple() const { return TypeTuple1(kind(), ref()); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    CompoundType(World& world, int kind, size_t num_elems);
    CompoundType(World& world, int kind, Elems elems);

    void dump_inner(Printer& printer) const;

public:

    TypeTupleN as_tuple() const { return TypeTupleN(kind(), elems()); }
    ANYDSL2_TYPE_HASH_EQUAL
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
    Sigma(World& world, const TypeTupleN& args)
        : CompoundType(world, args.get<0>(), args.get<1>())
        , named_(false)
    {}

public:

    bool named() const { return named_; }
    // TODO build setter for named sigmas which sets is_generic_

private:

    virtual void vdump(Printer& printer) const;
    virtual size_t hash() const;
    virtual bool equal(const Type* other) const;

    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public CompoundType {
private:

    Pi(World& world, const TypeTupleN& args)
        : CompoundType(world, args.get<0>(), args.get<1>())
    {}

public:

    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

typedef boost::tuple<int, size_t> GenericTuple;

class Generic : public Type {
private:

    Generic(World& world, const GenericTuple& args)
        : Type(world, args.get<0>(), 0, true)
        , index_(args.get<1>())
    {}
    ANYDSL2_TYPE_HASH_EQUAL

public:

    size_t index() const { return index_; }
    GenericTuple as_tuple() const { return GenericTuple(kind(), index()); }
    virtual void vdump(Printer& printer) const;

private:

    size_t index_;

    friend class World;
};

//------------------------------------------------------------------------------

typedef boost::tuple< int, ArrayRef<const Type*>, ArrayRef<uint32_t> > OpaqueTuple;

class Opaque : public CompoundType {
private:

    Opaque(World& world, const OpaqueTuple& args)
        : CompoundType(world, args.get<0>(), args.get<1>())
        , flags_(args.get<2>())
    {}

public:

    ArrayRef<uint32_t> flags() const { return flags_; }
    uint32_t flag(size_t i) const { return flags_[i]; }
    size_t num_flags() const { return flags_.size(); }
    OpaqueTuple as_tuple() const { return OpaqueTuple(kind(), elems(), flags()); }
    ANYDSL2_TYPE_HASH_EQUAL

private:

    virtual void vdump(Printer& printer) const;

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
