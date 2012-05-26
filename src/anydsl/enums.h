#ifndef ANYDSL_ENUMS_H
#define ANYDSL_ENUMS_H

#include "anydsl/util/types.h"

namespace anydsl {

//------------------------------------------------------------------------------

#define ANYDSL_GLUE(pre, next) \
    End_##pre, \
    Begin_##next = End_##pre, \
    zzz##Begin_##next = Begin_##next - 1,

enum IndexKind {
    Begin_Node = 0,
    __Begin_Node = -1,
#define ANYDSL_AIR_NODE(node) Index_##node,
#include "anydsl/tables/nodetable.h"

    ANYDSL_GLUE(Node, PrimType_u)

#define ANYDSL_U_TYPE(T) Index_PrimType_##T,
#include "anydsl/tables/primtypetable.h"

    ANYDSL_GLUE(PrimType_u, PrimType_f)

#define ANYDSL_F_TYPE(T) Index_PrimType_##T,
#include "anydsl/tables/primtypetable.h"

    ANYDSL_GLUE(PrimType_f, PrimLit_u)

#define ANYDSL_U_TYPE(T) Index_PrimLit_##T,
#include "anydsl/tables/primtypetable.h"

    ANYDSL_GLUE(PrimLit_u, PrimLit_f)

#define ANYDSL_F_TYPE(T) Index_PrimLit_##T,
#include "anydsl/tables/primtypetable.h"

    ANYDSL_GLUE(PrimLit_f, ArithOp)

#define ANYDSL_ARITHOP(op) Index_##op,
#include "anydsl/tables/arithoptable.h"

    ANYDSL_GLUE(ArithOp, RelOp)

#define ANYDSL_RELOP(op) Index_##op,
#include "anydsl/tables/reloptable.h"

    ANYDSL_GLUE(RelOp, ConvOp)

#define ANYDSL_CONVOP(op) Index_##op,
#include "anydsl/tables/convoptable.h"
    End_ConvOp,

    Begin_PrimType  = Begin_PrimType_u,
    Begin_PrimLit   = Begin_PrimLit_u,

    End_PrimType    = End_PrimType_f,
    End_PrimLit     = End_PrimLit_f,

    Num_Indexes     = End_ConvOp,

    Num_Nodes       = End_Node       - Begin_Node,

    Num_PrimTypes_u = End_PrimType_u - Begin_PrimType_u,
    Num_PrimTypes_f = End_PrimType_f - Begin_PrimType_f,
    Num_PrimLits_u  = End_PrimLit_u  - Begin_PrimLit_u,
    Num_PrimLits_f  = End_PrimLit_f  - Begin_PrimLit_f,

    Num_ArithOps    = End_ArithOp    - Begin_ArithOp,
    Num_RelOps      = End_RelOp      - Begin_RelOp,
    Num_ConvOps     = End_ConvOp     - Begin_ConvOp,

    Num_PrimTypes = Num_PrimTypes_u + Num_PrimTypes_f,
    Num_PrimLits  = Num_PrimTypes,
};

enum PrimTypeKind {
#define ANYDSL_U_TYPE(T) PrimType_##T = Index_PrimType_##T,
#define ANYDSL_F_TYPE(T) PrimType_##T = Index_PrimType_##T,
#include "anydsl/tables/primtypetable.h"
};

enum PrimLitKind {
#define ANYDSL_U_TYPE(T) PrimLit_##T = Index_PrimLit_##T,
#define ANYDSL_F_TYPE(T) PrimLit_##T = Index_PrimLit_##T,
#include "anydsl/tables/primtypetable.h"
};

enum PrimOpKind {
#define ANYDSL_ARITHOP(op) PrimOp_##op = Index_##op,
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op) PrimOp_##op = Index_##op,
#include "anydsl/tables/reloptable.h"

#define ANYDSL_CONVOP(op) PrimOp_##op = Index_##op,
#include "anydsl/tables/convoptable.h"
};

enum ArithOpKind {
#define ANYDSL_ARITHOP(op) ArithOp_##op = Index_##op,
#include "anydsl/tables/arithoptable.h"
};

enum RelOpKind {
#define ANYDSL_RELOP(op) RelOp_##op = Index_##op,
#include "anydsl/tables/reloptable.h"
};

enum ConvOpKind {
#define ANYDSL_CONVOP(op) ConvOp_##op = Index_##op,
#include "anydsl/tables/convoptable.h"
};

inline PrimTypeKind lit2type(PrimLitKind kind) {
    // it holds: Begin_PrimLit + offset = Begin_PrimType
    int offset = Begin_PrimType - Begin_PrimLit;

    // it holds: primLit + offset = primType
    return (PrimTypeKind) (((int) kind) + offset);
}

inline PrimLitKind type2lit(PrimTypeKind kind) {
    // it holds: Begin_PrimType + offset = Begin_PrimLit
    int offset = Begin_PrimLit - Begin_PrimType;

    // it holds: primType + offset = primLit
    return (PrimLitKind) (((int) kind) + offset);
}

inline bool isInteger(PrimTypeKind kind) {
    return (int) Begin_PrimType_u <= (int) kind && (int) kind < (int) End_PrimType_u;
}

inline bool isFloat(PrimTypeKind kind) {
    return (int) Begin_PrimType_f <= (int) kind && (int) kind < (int) End_PrimType_f;
}

bool isArithOp(IndexKind kind);
bool isRelOp(IndexKind kind);
bool isConvOp(IndexKind kind);

template<PrimTypeKind kind> struct kind2type {};
#define ANYDSL_U_TYPE(T) template<> struct kind2type<PrimType_##T> { typedef T type; };
#define ANYDSL_F_TYPE(T) template<> struct kind2type<PrimType_##T> { typedef T type; };
#include "anydsl/tables/primtypetable.h"

template<class T> struct type2kind {};
template<> struct type2kind<bool> { static const PrimTypeKind kind = PrimType_u1; };
#define ANYDSL_U_TYPE(T) template<> struct type2kind<T> { static const PrimTypeKind kind = PrimType_##T; };
#define ANYDSL_F_TYPE(T) template<> struct type2kind<T> { static const PrimTypeKind kind = PrimType_##T; };
#include "anydsl/tables/primtypetable.h"

const char* kind2str(PrimTypeKind kind);

} // namespace anydsl

#endif
