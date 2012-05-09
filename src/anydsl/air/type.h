#ifndef ANYDSL_TYPE_H
#define ANYDSL_TYPE_H

#include "anydsl/air/def.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Type : public Def {
protected:

    Type(IndexKind indexKind, const std::string& debug)
        : Def(indexKind, debug)
    {}
};

//------------------------------------------------------------------------------

class PrimType : public Type {
public:

    PrimType(PrimTypeKind primTypeKind, const std::string& debug = "")
        : Type((IndexKind) primTypeKind, debug)
    {}
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_TYPE_H
