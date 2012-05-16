#ifndef ANYDSL_AIR_TYPE_H
#define ANYDSL_AIR_TYPE_H

#include "anydsl/air/def.h"

namespace anydsl {

class PrimConst;
class World;

//------------------------------------------------------------------------------

class Type : public AIRNode {
protected:

    Type(World& world, IndexKind index, const std::string& debug)
        : AIRNode(index, debug)
        , world_(world)
    {}
    virtual ~Type() {}

public:

    World& world() const { return world_; }

private:

    World& world_;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind primTypeKind, const std::string& debug = "")
        : Type(world, (IndexKind) primTypeKind, debug)
    {}

public:

    virtual uint64_t hash() const { return (uint64_t) index(); }

    friend class World;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public Type {
public:

    Sigma(World& world, const std::string& debug)
        : Type(world, Index_Sigma, debug)
    {}

    /// Get element type via index.
    const Type* get(size_t) const { return 0; }

    /// Get element type via anydsl::PrimConst which serves as index.
    const Type* get(PrimConst*) const { return 0; }
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public Type {
public:

    Pi(World& world, const std::string& debug)
        : Type(world, Index_Pi, debug)
    {}

    /// Get element type.
    const Type* get(size_t) const { return 0; }

    /// Get element type via anydsl::PrimConst which serves as index.
    const Type* get(PrimConst*) const { return 0; }
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TYPE_H
