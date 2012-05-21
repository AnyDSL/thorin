#ifndef ANYDSL_FOLD_H
#define ANYDSL_FOLD_H

#include "anydsl/air/enums.h"
#include "anydsl/util/box.h"

namespace anydsl {

struct FoldRes {
    PrimTypeKind type;
    Box value;
    bool error;
};

FoldRes fold_bin(IndexKind kind, PrimTypeKind type, Box a, Box b);

Box fold_arith(ArithOpKind, PrimTypeKind type, Box a, Box b, bool& error);
Box fold_rel  (  RelOpKind, PrimTypeKind type, Box a, Box b, bool& error);

} // namespace anydsl

#endif // ANYDSL_FOLD_H
