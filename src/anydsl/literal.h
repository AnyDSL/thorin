#ifndef ANYDSL_LITERAL_H
#define ANYDSL_LITERAL_H

#include <vector>

#include "anydsl/primop.h"
#include "anydsl/type.h"
#include "anydsl/util/box.h"

namespace anydsl {

class Type;
class World;

//------------------------------------------------------------------------------

class Literal : public PrimOp {
protected:

    Literal(IndexKind index, const Type* type)
        : PrimOp(index, type, 0)
    {}
};

//------------------------------------------------------------------------------

class Undef : public Literal {
private:

    Undef(const Type* type)
        : Literal(Index_Undef, type)
    {}

    virtual void vdump(Printer& printer) const ;

    friend class World;

};

//------------------------------------------------------------------------------

class Error : public Literal {
private:

    Error(const Type* type)
        : Literal(Index_Error, type)
    {}

    virtual void vdump(Printer& printer) const ;

    friend class World;
};

//------------------------------------------------------------------------------

class PrimLit : public Literal {
private:

    PrimLit(const Type* type, Box box)
        : Literal((IndexKind) type2lit(type->as<PrimType>()->kind()), type)
        , box_(box)
    {}

public:

    PrimLitKind kind() const { return (PrimLitKind) indexKind(); }
    Box box() const { return box_; }

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

private:

    virtual void vdump(Printer& printer) const ;

    Box box_;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
