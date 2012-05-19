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
private:

    Undef(const Type* type)
        : Literal(Index_Undef, type)
    {}

public:

    virtual uint64_t hash() const { return hash(type()); }
    static  uint64_t hash(const Type*);

    friend class World;
};

//------------------------------------------------------------------------------

class ErrorLit : public Literal {
private:

    ErrorLit(const Type* type)
        : Literal(Index_ErrorLit, type)
    {}

public:

    virtual uint64_t hash() const { return hash(type()); }
    static  uint64_t hash(const Type*);

    friend class World;
};

//------------------------------------------------------------------------------

class PrimLit : public Literal {
private:

    PrimLit(World& world, PrimLitKind kind, Box box);

public:

    PrimLitKind kind() const { return (PrimLitKind) index(); }
    Box box() const { return box_; }

    virtual uint64_t hash() const { return hash(kind(), box()); }
    static  uint64_t hash(PrimLitKind kind, Box box);

private:

    Box box_;

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public Literal {
public:

    const Literals& elems() const { return elems_; }

    virtual uint64_t hash() const { return hash(type()); }
    static  uint64_t hash(const Type*) { return 0; }

private:

    Literals elems_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_LITERAL_H
