#ifndef ANYDSL_FOLD_H
#define ANYDSL_FOLD_H

#include "anydsl/enums.h"
#include "anydsl/util/box.h"

namespace anydsl {

struct FoldValue {
    enum Kind {
        Valid,
        Error,
        Undef,
    };

    Kind kind;
    PrimTypeKind type;
    Box box;


    FoldValue(PrimTypeKind type)
        : kind(Valid)
        , type(type)
    {}
    FoldValue(PrimTypeKind type, Box box)
        : kind(Valid)
        , type(type)
        , box(box)
    {}

};

FoldValue fold_bin(IndexKind kind, PrimTypeKind type, FoldValue a, FoldValue b);

} // namespace anydsl

#endif // ANYDSL_FOLD_H
