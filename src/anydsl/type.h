#ifndef ANYDSL_TYPE_H
#define ANYDSL_TYPE_H

#include <iterator>

#include "anydsl/def.h"
#include "anydsl/util/array.h"

namespace anydsl {

class PrimLit;
class Pi;
class World;

typedef boost::unordered_set<const Def*> Instances;

//------------------------------------------------------------------------------

class Type : public Def {
protected:

    Type(World& world, int kind, size_t num)
        : Def(kind, 0, num)
        , world_(world)
    {}
    virtual ~Type();

public:

    World& world() const { return world_; }
    const Instances& instances() const { return instances_; }

private:

    void registerInstance(const Def* def) const;
    void unregisterInstance(const Def* def) const;

    World& world_;
    mutable Instances instances_;

    friend class Def;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind kind);

public:

    PrimTypeKind primtype_kind() const { return (PrimTypeKind) node_kind(); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

inline const Type* const& elem_as_type(const Def* const* ptr) { 
    assert((*ptr)->as<Type>());
    return *((const Type* const*) ptr); 
}

class CompoundType : public Type {
protected:

    CompoundType(World& world, int kind, size_t num);
    CompoundType(World& world, int kind, ArrayRef<const Type*> elems);

public:

    /// Get element type via index.
    const Type* elem(size_t i) const { 
        anydsl_assert(i < elems().size(), "index out of range"); 
        return op(i)->as<Type>();
    }

    typedef ArrayRef<const Def*, const Type*, elem_as_type> Elems;
    Elems elems() const { return Elems(Def::ops().begin().base(), Def::ops().size()); }
    size_t numelems() const { return numops(); }

protected:

    void dumpInner(Printer& printer) const;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, size_t num)
        : CompoundType(world, Node_Sigma, num)
        , named_(true)
    {}
    Sigma(World& world, ArrayRef<const Type*> elems)
        : CompoundType(world, Node_Sigma, elems)
        , named_(false)
    {}

public:

    bool named() const { return named_; }

private:

    virtual void vdump(Printer& printer) const;
    virtual size_t hash() const;
    virtual bool equal(const Def* other) const;

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

    static const size_t npos = -1;
    size_t nextPi(size_t pos = 0) const;
    bool isHigherOrder() const { return nextPi() != npos; }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
