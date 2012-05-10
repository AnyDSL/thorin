#ifndef ANYDSL_AIR_TYPE_H
#define ANYDSL_AIR_TYPE_H

#include "anydsl/air/def.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Type : public Def {
protected:

    Type(PrimTypeKind primTypeKind, const std::string& debug)
        : Def((IndexKind) primTypeKind, debug)
    {}
};

//------------------------------------------------------------------------------

class PrimType : public Type {
public:

    PrimType(PrimTypeKind primTypeKind, const std::string& debug = "")
        : Type(primTypeKind, debug)
    {}

    virtual uint64_t hash() const { return (uint64_t) indexKind(); }
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_TYPE_H
