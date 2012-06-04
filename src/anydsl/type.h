#ifndef ANYDSL_TYPE_H
#define ANYDSL_TYPE_H

#include "anydsl/defuse.h"

namespace anydsl {

class PrimLit;
class Pi;
class World;

//------------------------------------------------------------------------------

class Type : public AIRNode {
protected:

    Type(World& world, IndexKind index)
        : AIRNode(index)
        , world_(world)
    {}

public:

    World& world() const { return world_; }

    virtual bool equal(const Type* other) const;
    virtual size_t hash() const;

private:

    World& world_;
};

typedef std::vector<const Type*> Types;

//------------------------------------------------------------------------------

class NoRet : public Type {
private:

    NoRet(World& world, const Pi* pi)
        : Type(world, Index_NoRet)
        , pi_(pi)
    {}

public:

    const Pi* pi() const { return pi_; }

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

class CompoundType : public Type {
protected:

    /// Copies over the range specified by \p begin and \p end.
    template<class T>
    CompoundType(World& world, IndexKind index, T begin, T end) 
        : Type(world, index)
    {
        for (T i = begin; i != end; ++i)
           types_.push_back(*i);
    }

public:

    /// Get element type via index.
    const Type* get(size_t i) const { 
        anydsl_assert(i < types_.size(), "index out of range"); 
        return types_[i]; 
    }

    /// Get element type via anydsl::PrimLit which serves as index.
    const Type* get(PrimLit* i) const;

    const Types& types() const { return types_; }

    virtual bool equal(const Type* other) const;
    virtual size_t hash() const;

protected:

    Types types_;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world)
        : CompoundType(world, Index_Sigma, (const Type**) 0, (const Type**) 0)
        , named_(true)
    {}

    template<class T>
    Sigma(World& world, T begin, T end)
        : CompoundType(world, Index_Sigma, begin, end)
        , named_(false)
    {}

public:

    bool named() const { return named_; }

    template<class T>
    void set(T begin, T end) {
        anydsl_assert(named_, "only allowed on named Sigmas");
        anydsl_assert(types_.empty(), "members already set");
        types_.insert(types_.begin(), begin, end);
    }

private:

    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public CompoundType {
private:

    template<class T>
    Pi(World& world, T begin, T end)
        : CompoundType(world, Index_Pi, begin, end)
    {}


    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
