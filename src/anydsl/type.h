#ifndef ANYDSL_TYPE_H
#define ANYDSL_TYPE_H

#include <iterator>

#include "anydsl/def.h"

namespace anydsl {

class PrimLit;
class Pi;
class World;

//------------------------------------------------------------------------------

class Type : public Value {
protected:

    Type(World& world, IndexKind index, size_t num)
        : Value(index, 0, num)
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

    PrimTypeKind kind() const { return (PrimTypeKind) index(); }

    friend class World;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public Type {
private:

    Sigma(World& world, size_t num);
    Sigma(World& world, const Type* const* begin, const Type* const* end);

public:

    bool named() const { return named_; }

    /// Get element type via index.
    const Type* get(size_t i) const { 
        anydsl_assert(i < numOps(), "index out of range"); 
        return ops_[i]->as<Type>();
    }

    /// Get element type via anydsl::PrimLit which serves as index.
    const Type* get(const PrimLit* i) const;

private:

    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public Type {
private:

    Pi(const Sigma* sigma);

public:

    /// Get element type via index.
    const Type* get(size_t i) const { return sigma()->get(i); }
    /// Get element type via anydsl::PrimLit which serves as index.
    const Type* get(PrimLit* i) const { return sigma()->get(i); }

    const Sigma* sigma() const;

private:

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
