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

    Undef(const ValueNumber& vn)
        : Literal((IndexKind) vn.index, (const Type*) vn.op1)
    {}

    static ValueNumber VN(const Type* type) { return ValueNumber(Index_Undef, (uintptr_t) type); }

    friend class World;
};

//------------------------------------------------------------------------------

class ErrorLit : public Literal {
private:

    ErrorLit(const ValueNumber& vn)
        : Literal((IndexKind) vn.index, (const Type*) vn.op1)
    {}

    static ValueNumber VN(const Type* type) { return ValueNumber(Index_ErrorLit, (uintptr_t) type); }

    friend class World;
};

//------------------------------------------------------------------------------

class PrimLit : public Literal {
private:

    PrimLit(const ValueNumber& vn);

public:

    PrimLitKind kind() const { return (PrimLitKind) index(); }
    Box box() const { return box_; }


    static ValueNumber VN(const Type* type, Box box);

private:

    struct Split { 
        uintptr_t op1;
        uintptr_t op2;
    };

    Box box_;

    friend class World;
};

//------------------------------------------------------------------------------

class Tuple : public Literal {
public:

    const Literals& elems() const { return elems_; }

private:

    Literals elems_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_LITERAL_H
