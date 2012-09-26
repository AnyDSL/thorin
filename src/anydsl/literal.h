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

    Literal(int kind, const Type* type)
        : PrimOp(kind, type, 0)
    {}

protected:

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;
};

//------------------------------------------------------------------------------

/// Base class for \p Any and \p Bottom.
class Undef : public Literal {
protected:

    Undef(int kind, const Type* type)
        : Literal(kind, type)
    {}
};

//------------------------------------------------------------------------------

/** 
 * @brief The wish-you-a-value value.
 *
 * This literal represents an arbitrary value.
 * When ever an operation takes an \p Undef value as argument, 
 * you may literally wish your favorite value instead.
 */
class Any : public Undef {
private:

    Any(const Type* type)
        : Undef(Node_Undef, type)
    {}
    virtual Any* clone() const { return new Any(*this); }
    virtual void vdump(Printer& printer) const ;

    friend class World;
};

//------------------------------------------------------------------------------

/** 
 * @brief The novalue-value.
 *
 * This literal represents literally 'no value'.
 * Extremely useful for data flow analysis.
 */
class Bottom : public Undef {
private:

    Bottom(const Type* type)
        : Undef(Node_Bottom, type)
    {}
    virtual Bottom* clone() const { return new Bottom(*this); }
    virtual void vdump(Printer& printer) const ;

    friend class World;
};

//------------------------------------------------------------------------------

class PrimLit : public Literal {
private:

    PrimLit(const Type* type, Box box)
        : Literal(Node_PrimLit, type)
        , box_(box)
    {}
    PrimLit(const PrimLit& lit)
        : Literal(lit)
        , box_(lit.box())
    {}
    virtual PrimLit* clone() const { return new PrimLit(*this); }

public:

    Box box() const { return box_; }
    const PrimType* primtype() const { return type()->as<PrimType>(); }
    PrimTypeKind primtype_kind() const { return primtype()->primtype_kind(); }

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;

private:

    virtual void vdump(Printer& printer) const ;

    Box box_;

    friend class World;
};

//------------------------------------------------------------------------------

/**
 * The sole purpose of this node is to hold types.
 * This node is not destroyed by the dead code elimination, and hence,
 * the held type won't be destroyed in the unused type elimination.
 */
class TypeKeeper : public Literal {
private:

    TypeKeeper(const Type* type)
        : Literal(Node_TypeKeeper, type)
    {}
    virtual TypeKeeper* clone() const { return new TypeKeeper(*this); }
    virtual void vdump(Printer& printer) const ;

    friend class World;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
