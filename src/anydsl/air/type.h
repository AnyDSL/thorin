#ifndef ANYDSL_AIR_TYPE_H
#define ANYDSL_AIR_TYPE_H

#include "anydsl/air/def.h"

namespace anydsl {

class PrimConst;
class World;

//------------------------------------------------------------------------------

class Type : public AIRNode {
protected:

    Type(World& world, IndexKind index, const std::string& name)
        : AIRNode(index, name)
        , world_(world)
    {}
    virtual ~Type() {}

public:

    World& world() const { return world_; }

private:

    World& world_;
};

typedef std::vector<const Type*> Types;

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind primTypeKind);

public:

    virtual uint64_t hash() const { return (uint64_t) index(); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    /// Creates an empty \p CompoundType.
    CompoundType(World& world, IndexKind index, const std::string& name)
        : Type(world, index, name)
    {}

    /// Copies over the range specified by \p begin and \p end to \p types_.
    template<class T>
    CompoundType(World& world, IndexKind index, T begin, T end, const std::string& name)
        : Type(world, index, name)
    {
        types_.insert(types_.begin(), begin, end);
    }

public:

    /// Get element type via index.
    const Type* get(size_t i) const { 
        anydsl_assert(i < types_.size(), "index out of range"); 
        return types_[i]; 
    }

    /// Get element type via anydsl::PrimConst which serves as index.
    const Type* get(PrimConst* c) const;

private:

    Types types_;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, const std::string& name)
        : CompoundType(world, Index_Sigma, name)
    {}

    template<class T>
    Sigma(World& world, T begin, T end, const std::string& name)
        : CompoundType(world, Index_Sigma, begin, end, name)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public CompoundType {
private:

    Pi(World& world, const std::string& name)
        : CompoundType(world, Index_Pi, name)
    {}

    template<class T>
    Pi(World& world, T begin, T end, const std::string& name)
        : CompoundType(world, Index_Sigma, begin, end, name)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TYPE_H
