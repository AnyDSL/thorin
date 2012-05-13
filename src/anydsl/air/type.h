#ifndef ANYDSL_AIR_TYPE_H
#define ANYDSL_AIR_TYPE_H

#include "anydsl/air/def.h"

namespace anydsl {

class PrimConst;
class World;

//------------------------------------------------------------------------------

class Type : public AIRNode {
protected:

    Type(World& world, PrimTypeKind primTypeKind, const std::string& debug)
        : AIRNode((IndexKind) primTypeKind, debug)
        , world_(world)
    {}
    virtual ~Type() {}

public:

    World& world() const { return world_; }

private:

    World& world_;
};

//------------------------------------------------------------------------------

class PrimType : public Type {
private:

    PrimType(World& world, PrimTypeKind primTypeKind, const std::string& debug = "")
        : Type(world, primTypeKind, debug)
    {}

public:

    virtual uint64_t hash() const { return (uint64_t) index(); }

    friend class World;
};

//------------------------------------------------------------------------------

class Sigma : public Type {
public:

    Type* get(size_t) { return 0; }
    Type* get(PrimConst*) { return 0; }
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TYPE_H
