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

private:

    World& world_;
};

typedef std::vector<const Type*> Types;

//------------------------------------------------------------------------------

class NoRet : public Type {
private:

    NoRet(World& world, const ValueNumber& vn)
        : Type(world, vn.index)
        , pi_((const Pi*) vn.op1)
    {}

    static ValueNumber VN(const Pi* pi) { return ValueNumber(Index_NoRet, uintptr_t(pi)); }

    const Pi* pi_;

public:

    const Pi* pi() const { return pi_; }

    friend class World;
};

//------------------------------------------------------------------------------

/// Primitive types -- also known as atomic or scalar types.
class PrimType : public Type {
private:

    PrimType(World& world, const ValueNumber& vn);

    static ValueNumber VN(PrimTypeKind kind) { return ValueNumber((IndexKind) kind); }

public:

    PrimTypeKind kind() const { return (PrimTypeKind) index(); }

    friend class World;
};

//------------------------------------------------------------------------------

class CompoundType : public Type {
protected:

    /// Only used by Sigma to create named Sigma%s.
    CompoundType(World& world, IndexKind index)
        : Type(world, index)
    {}

    CompoundType(World& world, const ValueNumber& vn)
        : Type(world, vn.index)
    {
        for (size_t i = 0, e = vn.size; i != e; ++i)
            types_.push_back((const Type*) vn.more[i]);
    }

    /// Copies over the range specified by \p begin and \p end.
    template<class T>
    static ValueNumber VN(IndexKind index, T begin, T end) {
        size_t size = std::distance(begin, end);
        ValueNumber vn = ValueNumber::createMore(index, size);
        size_t x = 0;
        for (T i = begin; i != end; ++i, ++x)
            vn.more[x] = uintptr_t(*i);

        return vn;
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

    template<class T>
    static size_t hash(T begin, T end) {  return 0; }

protected:

    Types types_;
};

//------------------------------------------------------------------------------

/// A tuple type.
class Sigma : public CompoundType {
private:

    Sigma(World& world)
        : CompoundType(world, Index_Sigma)
        , named_(true)
    {}

    Sigma(World& world, const ValueNumber& vn)
        : CompoundType(world, vn)
        , named_(false)
    {}

    template<class T>
    static ValueNumber VN(T begin, T end) {
        return CompoundType::VN(Index_Sigma, begin, end);
    }

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

    Pi(World& world, const ValueNumber& vn)
        : CompoundType(world, vn)
    {}


    template<class T>
    static ValueNumber VN(T begin, T end) {
        return CompoundType::VN(Index_Pi, begin, end);
    }

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
