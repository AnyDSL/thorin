#ifndef ANYDSL_AIR_LITERAL_H
#define ANYDSL_AIR_LITERAL_H

#include <vector>

#include "anydsl/air/def.h"
#include "anydsl/util/box.h"

namespace anydsl {

class Type;
class World;

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

} // namespace anydsl

#endif // ANYDSL_AIR_LITERAL_H
