#ifndef ANYDSL_AIR_ENUMS_H
#define ANYDSL_AIR_ENUMS_H

namespace anydsl {

//------------------------------------------------------------------------------

#define ANYDSL_GLUE(pre, next) \
    End_##pre, \
    Begin_##next = End_##pre, \
    __##Begin_##next = Begin_##next - 1,

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

    ANYDSL_GLUE(PrimType_f, PrimConst_u)

#define ANYDSL_U_TYPE(T) Index_PrimConst_##T,
#include "anydsl/tables/primtypetable.h"

    ANYDSL_GLUE(PrimConst_u, PrimConst_f)

#define ANYDSL_F_TYPE(T) Index_PrimConst_##T,
#include "anydsl/tables/primtypetable.h"

    ANYDSL_GLUE(PrimConst_f, ArithOp)

#define ANYDSL_ARITHOP(op) Index_##op,
#include "anydsl/tables/arithoptable.h"

    ANYDSL_GLUE(ArithOp, RelOp)

#define ANYDSL_RELOP(op) Index_##op,
#include "anydsl/tables/reloptable.h"

    ANYDSL_GLUE(RelOp, ConvOp)

#define ANYDSL_CONVOP(op) Index_##op,
#include "anydsl/tables/convoptable.h"
    End_ConvOp,

    Num_Indexes = End_ConvOp,

    Num_Nodes        = End_Node        - Begin_Node,

    Num_PrimTypes_u  = End_PrimType_u  - Begin_PrimType_u,
    Num_PrimTypes_f  = End_PrimType_f  - Begin_PrimType_f,
    Num_PrimConsts_u = End_PrimConst_u - Begin_PrimConst_u,
    Num_PrimConsts_f = End_PrimConst_f - Begin_PrimConst_f,

    Num_ArithOps     = End_ArithOp     - Begin_ArithOp,
    Num_RelOps       = End_RelOp       - Begin_RelOp,
    Num_ConvOps      = End_ConvOp      - Begin_ConvOp,

    Num_PrimTypes = Num_PrimTypes_u + Num_PrimTypes_f,
    Num_PrimConsts = Num_PrimTypes,
};

enum NodeKind {
#define ANYDSL_AIR_NODE(node) Node_##node = Index_##node,
#include "anydsl/tables/nodetable.h"
};

enum PrimTypeKind {
#define ANYDSL_U_TYPE(T) PrimType_##T = Index_PrimType_##T,
#define ANYDSL_F_TYPE(T) PrimType_##T = Index_PrimType_##T,
#include "anydsl/tables/primtypetable.h"
};

enum PrimConstKind {
#define ANYDSL_U_TYPE(T) Prim_##T = Index_PrimConst_##T,
#define ANYDSL_F_TYPE(T) Prim_##T = Index_PrimConst_##T,
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

} // namespace anydsl

#endif // ANYDSL_AIR_ENUMS_H
