#ifndef ANYDSL_ENUMS_H
#define ANYDSL_ENUMS_H

namespace anydsl {

enum Nodekind {
#define ANYDSL_AIR_NODE(node) Kind_##node,
#include "anydsl/tables/airnodetable.h"
};

enum ArithOpKind {
#define ANYDSL_ARITHOP(op) Kind_##op,
#include "anydsl/tables/arithoptable.h"
};

enum RelOpKind {
#define ANYDSL_RELOP(op) Kind_##op,
#include "anydsl/tables/reloptable.h"
};

enum ConvOpKind {
#define ANYDSL_CONVOP(op) Kind_##op,
#include "anydsl/tables/convoptable.h"
};

enum PrimitiveTypes {
#define ANYDSL_U_TYPE(T) Kind_##T,
#define ANYDSL_F_TYPE(T) Kind_##T,
#include "anydsl/tables/typetable.h"
};

} // namespace anydsl

#endif // ANYDSL_ENUMS_H
