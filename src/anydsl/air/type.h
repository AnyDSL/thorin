#ifndef ANYDSL_AIR_TYPE_H
#define ANYDSL_AIR_TYPE_H

#include "anydsl/air/def.h"

namespace anydsl {

class PrimLit;
class World;

//------------------------------------------------------------------------------

class Type : public AIRNode {
protected:

    Type(World& world, IndexKind index)
        : AIRNode(index)
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

class ErrorType : public Type {
private:

    ErrorType(World& world) : Type(world, Index_ErrorType) {}

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

class CompoundType : public Type {
protected:

    /// Creates an empty \p CompoundType.
    CompoundType(World& world, IndexKind index)
        : Type(world, index)
    {}

    /// Copies over the range specified by \p begin and \p end to \p types_.
    template<class T>
    CompoundType(World& world, IndexKind index, T begin, T end)
        : Type(world, index)
    {
        types_.insert(types_.begin(), begin, end);
    }

public:

    /// Get element type via index.
    const Type* get(size_t i) const { 
        anydsl_assert(i < types_.size(), "index out of range"); 
        return types_[i]; 
    }

    template<class T>
    bool equal(T begin, T end) {
        bool result = true;
        Types::const_iterator j = types_.begin(), je = types_.end();
        for (T i = begin, ie = end; i != ie && j != je && result; ++i, ++j)
            if (*i != *j)
                return false;
        return true;
    }

    /// Get element type via anydsl::PrimLit which serves as index.
    const Type* get(PrimLit* i) const;

    const Types& types() const { return types_; }

    template<class T>
    static size_t hash(T begin, T end) {  return 0; }

protected:

    Types types_;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world, bool named = false)
        : CompoundType(world, Index_Sigma)
        , named_(named)
    {}

    /// Creates an unamed Sigma from the given range.
    template<class T>
    Sigma(World& world, T begin, T end, bool named = false)
        : CompoundType(world, Index_Sigma, begin, end)
        , named_(named)
    {}

public:

    bool named() const { return named_; }

    template<class T>
    void set(T begin, T end) {
        anydsl_assert(named_, "only allowed on named Sigmas");
        anydsl_assert(types_.empty(), "members already set");
        types_.insert(types_.begin(), begin, end);
    }

    template<class T>
    static uint64_t hash(T begin, T end) { return 0; }

private:

    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public CompoundType {
private:

    Pi(World& world)
        : CompoundType(world, Index_Pi)
    {}

    template<class T>
    Pi(World& world, T begin, T end)
        : CompoundType(world, Index_Sigma, begin, end)
    {}

    template<class T>
    static uint64_t hash(T begin, T end) { return 0; }

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TYPE_H
