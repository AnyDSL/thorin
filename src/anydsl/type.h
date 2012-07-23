#ifndef ANYDSL_TYPE_H
#define ANYDSL_TYPE_H

#include <iterator>

#include "anydsl/def.h"
#include "anydsl/util/arrayref.h"

namespace anydsl {

class PrimLit;
class Pi;
class World;

//------------------------------------------------------------------------------

class Type : public Def {
protected:

    Type(World& world, int kind, size_t num)
        : Def(kind, 0, num)
        , world_(world)
    {}

public:

    World& world() const { return world_; }

private:

    World& world_;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind kind);

public:

    PrimTypeKind kind() const { return (PrimTypeKind) indexKind(); }

private:

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

inline const Type*& elem_as_type(const Def** ptr) { 
    assert((*ptr)->as<Type>());
    return *((const Type**) ptr); 
}

class CompoundType : public Type {
protected:

    CompoundType(World& world, IndexKind index, size_t num);
    CompoundType(World& world, IndexKind index, ArrayRef<const Type*> elems);

public:

    /// Get element type via index.
    const Type* get(size_t i) const { 
        anydsl_assert(i < numOps(), "index out of range"); 
        return op(i)->as<Type>();
    }

    typedef ArrayRef<const Def*, const Type*, elem_as_type> Elems;
    Elems elems() const { return polyOps<Elems>(); }
    size_t numElems() const { return numOps(); }

protected:

    void dumpInner(Printer& printer) const;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, size_t num)
        : CompoundType(world, Index_Sigma, num)
        , named_(true)
    {}
    Sigma(World& world, ArrayRef<const Type*> elems)
        : CompoundType(world, Index_Sigma, elems)
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
        : CompoundType(world, Index_Pi, elems)
    {}

    virtual void vdump(Printer& printer) const;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
