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

struct TypeHash : std::unary_function<const Type*, size_t> {
    size_t operator () (const Type* t) const { return t->hash(); }
};

struct TypeEqual : std::binary_function<const Type*, const Type*, bool> {
    bool operator () (const Type* t1, const Type* t2) const { return t1->equal(t2); }
};

typedef std::vector<const Type*> Types;
typedef std::pair<const Type* const*, const Type* const*> TypeRange;

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

/// A tuple type.
class Sigma : public Type {
private:

    Sigma(World& world)
        : Type(world, Index_Sigma)
        , named_(true)
    {}

    Sigma(World& world, const Type* const* begin, const Type* const* end)
        : Type(world, Index_Sigma)
        , named_(false)
    {
        for (const Type* const* i = begin; i != end; ++i)
           types_.push_back(*i);
    }


public:

    bool named() const { return named_; }

    template<class T>
    void set(T begin, T end) {
        anydsl_assert(named_, "only allowed on named Sigmas");
        anydsl_assert(types_.empty(), "members already set");
        types_.insert(types_.begin(), begin, end);
    }

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

private:

    Types types_;
    bool named_;

    friend class World;
};

//------------------------------------------------------------------------------

/// A function type.
class Pi : public Type {
private:

    Pi(const Sigma* sigma)
        : Type(sigma->world(), Index_Pi)
        , sigma_(sigma)
    {
        anydsl_assert(!sigma->named(), "only unnamed sigma allowed with pi type");
    }

public:

    const Type* sigma() const { return sigma_; }

    /// Get element type via index.
    const Type* get(size_t i) const { return sigma_->get(i); }
    /// Get element type via anydsl::PrimLit which serves as index.
    const Type* get(PrimLit* i) const { return sigma_->get(i); }

    const Types& types() const { return sigma_->types(); }

    virtual bool equal(const Type* other) const;
    virtual size_t hash() const;

private:

    const Sigma* sigma_;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
