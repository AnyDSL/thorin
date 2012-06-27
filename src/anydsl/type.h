#ifndef ANYDSL_TYPE_H
#define ANYDSL_TYPE_H

#include <iterator>

#include "anydsl/def.h"

namespace anydsl {

class PrimLit;
class Pi;
class World;

//------------------------------------------------------------------------------

class Type : public Def {
protected:

    Type(World& world, IndexKind index, size_t num)
        : Def(index, 0, num)
        , world_(world)
    {}

public:

    World& world() const { return world_; }

private:

    World& world_;
};

//------------------------------------------------------------------------------

class NoRet : public Type {
private:

    NoRet(World& world, const Pi* pi);

public:

    const Pi* pi() const { return ops_[0]->as<Pi>(); }

private:

    const Pi* pi_;

    friend class World;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind kind);

public:

    PrimTypeKind kind() const { return (PrimTypeKind) indexKind(); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    CompoundType(World& world, IndexKind index, size_t num);
    CompoundType(World& world, IndexKind index, const Type* const* begin, const Type* const* end);

public:

    /// Get element type via index.
    const Type* get(size_t i) const { 
        anydsl_assert(i < numOps(), "index out of range"); 
        return ops_[i]->as<Type>();
    }

    /// Get element type via anydsl::PrimLit which serves as index.
    const Type* get(const PrimLit* i) const;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, size_t num)
        : CompoundType(world, Index_Sigma, num)
        , named_(true)
    {}
    Sigma(World& world, const Type* const* begin, const Type* const* end)
        : CompoundType(world, Index_Sigma, begin, end)
        , named_(false)
    {}

public:

    bool named() const { return named_; }

private:

    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public CompoundType {
private:

    Pi(World& world, const Type* const* begin, const Type* const* end)
        : CompoundType(world, Index_Pi, begin, end)
    {}

public:

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
