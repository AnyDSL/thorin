#ifndef ANYDSL_AIR_TYPE_H
#define ANYDSL_AIR_TYPE_H

#include "anydsl/air/def.h"

namespace anydsl {

class Universe;

//------------------------------------------------------------------------------

class Type : public AIRNode {
protected:

    Type(Universe& universe, PrimTypeKind primTypeKind, const std::string& debug)
        : AIRNode((IndexKind) primTypeKind, debug)
        , universe_(universe)
    {}

public:

    Universe& universe() const { return universe_; }

private:

    Universe& universe_;
};

//------------------------------------------------------------------------------

class PrimType : public Type {
private:

    PrimType(Universe& universe, PrimTypeKind primTypeKind, const std::string& debug = "")
        : Type(universe, primTypeKind, debug)
    {}

public:

    virtual uint64_t hash() const { return (uint64_t) index(); }

    friend class Universe;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TYPE_H
