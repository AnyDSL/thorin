#ifndef THORIN_LITERAL_H
#define THORIN_LITERAL_H

#include <vector>

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/util/box.h"

namespace thorin {

class Type;
class World;

//------------------------------------------------------------------------------

class Literal : public PrimOp {
protected:
    Literal(NodeKind kind, const Type* type, const std::string& name)
        : PrimOp(0, kind, type, name)
    {}
};

//------------------------------------------------------------------------------

/// Base class for \p Any and \p Bottom.
class Undef : public Literal {
protected:
    Undef(NodeKind kind, const Type* type, const std::string& name)
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
#define THORIN_ALL_TYPE(T) T T##_value() const { return value().get_##T(); }
#include "thorin/tables/primtypetable.h"
    
    const PrimType* primtype() const { return type()->as<PrimType>(); }
    PrimTypeKind primtype_kind() const { return primtype()->primtype_kind(); }
    virtual size_t hash() const { return hash_combine(Literal::hash(), bcast<uint64_t, Box>(value())); }
    virtual bool equal(const PrimOp* other) const { 
        return Literal::equal(other) ? this->value() == other->as<PrimLit>()->value() : false; 
    }

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
T DefNode::primlit_value() const {
    const PrimLit* lit = this->as<PrimLit>();
    switch (lit->primtype_kind()) {
#define THORIN_ALL_TYPE(U) case PrimType_##U: return (T) lit->value().get_##U();
#include "thorin/tables/primtypetable.h"
        default: THORIN_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

} // namespace thorin

#endif
