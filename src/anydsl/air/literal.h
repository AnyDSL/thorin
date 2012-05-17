#ifndef ANYDSL_AIR_LITERAL_H
#define ANYDSL_AIR_LITERAL_H

#include <vector>

#include <boost/unordered_set.hpp>

#include "anydsl/air/def.h"
#include "anydsl/util/box.h"

namespace anydsl {

class Type;
class Terminator;
class World;

typedef boost::unordered_set<Lambda*> Fix;

//------------------------------------------------------------------------------

class Literal : public Value {
protected:

    Literal(IndexKind index, const Type* type)
        : Value(index, type)
    {}
};

typedef std::vector<Literal*> Literals;

//------------------------------------------------------------------------------

class Undef : public Literal {
public:

    Undef(const Type* type)
        : Literal(Index_Undef, type)
        {}
};

//------------------------------------------------------------------------------

class PrimLit : public Literal {
public:

    PrimLit(World& world, PrimTypeKind kind, Box box);

    PrimTypeKind kind() { return (PrimTypeKind) index(); }
    Box box() const { return box_; }

    virtual uint64_t hash() const;

private:

    Box box_;
};

//------------------------------------------------------------------------------

class Tuple : public Literal {
public:

    const Literals& elems() const { return elems_; }

    virtual uint64_t hash() const;

private:

    Literals elems_;
};

//------------------------------------------------------------------------------

class Lambda : public Literal {
private:

    /**
     * Use this constructor if you know the type beforehand.
     * You are still free to append other params later on.
     */
    Lambda(Lambda* parent, const Type* type);

    /**
     * Use this constructor if you want to incrementally build the type.
     * Initially the type is set to "pi()".
     */
    Lambda(World& world, Lambda* parent);
    ~Lambda();

public:

    const Fix& fix() const { return fix_; }
    const Fix& siblings() const { assert(parent_); return parent_->fix(); }

    Terminator* terminator() { return terminator_; }
    const Terminator* terminator() const { return terminator_; }

    void setTerminator(Terminator* Terminator) { terminator_ = Terminator; }

    void insert(Lambda* lambda);
    void remove(Lambda* lambda);

    virtual uint64_t hash() const { return 0; /* TODO */ }

private:

    Lambda* parent_;
    Terminator* terminator_;
    Fix fix_;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_LITERAL_H
