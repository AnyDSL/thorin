#ifndef ANYDSL2_LITERAL_H
#define ANYDSL2_LITERAL_H

#include <vector>

#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/util/box.h"

namespace anydsl2 {

class Type;
class World;

//------------------------------------------------------------------------------

class Literal : public PrimOp {
protected:

    Literal(int kind, const Type* type, const std::string& name)
        : PrimOp(0, kind, type, name)
    {}
};

//------------------------------------------------------------------------------

/// Base class for \p Any and \p Bottom.
class Undef : public Literal {
protected:

    Undef(int kind, const Type* type, const std::string& name)
        : Literal(kind, type, name)
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

    Any(const Type* type, const std::string& name)
        : Undef(Node_Any, type, name)
    {}

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

    Bottom(const Type* type, const std::string& name)
        : Undef(Node_Bottom, type, name)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

class PrimLit : public Literal {
private:

    PrimLit(World& world, PrimTypeKind kind, Box box, const std::string& name);

public:

    Box value() const { return box_; }
#define ANYDSL2_U_TYPE(T) \
    T T##_value() const { return value().get_##T(); }
#define ANYDSL2_F_TYPE(T) ANYDSL2_U_TYPE(T)
#include "anydsl2/tables/primtypetable.h"
    
    const PrimType* primtype() const { return type()->as<PrimType>(); }
    PrimTypeKind primtype_kind() const { return primtype()->primtype_kind(); }
    virtual size_t hash() const { return hash_combine(Literal::hash(), bcast<uint64_t, Box>(value())); }
    bool equal(const Node* other) const { return Literal::equal(other) ? this->value() == other->as<PrimLit>()->value() : false; }

private:

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

    TypeKeeper(const Type* type, const std::string& name)
        : Literal(Node_TypeKeeper, type, name)
    {}

    friend class World;
};

//------------------------------------------------------------------------------

template<class T>
T Def::primlit_value() const {
    const PrimLit* lit = this->as<PrimLit>();
    switch (lit->primtype_kind()) {
#define ANYDSL2_UF_TYPE(U) case PrimType_##U: return (T) lit->value().get_##U();
#include "anydsl2/tables/primtypetable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
