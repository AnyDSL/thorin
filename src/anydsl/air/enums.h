#ifndef ANYDSL_AIR_ENUMS_H
#define ANYDSL_AIR_ENUMS_H

namespace anydsl {

namespace detail {

enum CountPrimTypes_u {
#define ANYDSL_U_TYPE(T) CountType_u_##T,
#include "anydsl/tables/typetable.h"
    Num_PrimTypes_u,
};

enum CountPrimTypes_f {
#define ANYDSL_F_TYPE(T) CountType_f_##T,
#include "anydsl/tables/typetable.h"
    Num_PrimTypes_f,
};

enum CountNodes {
#define ANYDSL_AIR_NODE(node) CountNode_##node,
#include "anydsl/tables/airnodetable.h"
    Num_Nodes,
};

enum CountArithOps {
#define ANYDSL_ARITHOP(op) CountArithOp_##op,
#include "anydsl/tables/arithoptable.h"
    Num_ArithOps,
};

enum CountRelOps {
#define ANYDSL_RELOP(op) CountRelOp_##op,
#include "anydsl/tables/reloptable.h"
    Num_RelOps,
};

enum CountConvOps {
#define ANYDSL_CONVOP(op) CountConvOp_##op,
#include "anydsl/tables/convoptable.h"
    Num_ConvOps,
};

} // namespace detail

enum Counter {
    Num_PrimTypes_u = detail::Num_PrimTypes_u,
    Num_PrimTypes_f = detail::Num_PrimTypes_f,
    Num_PrimTypes   = detail::Num_PrimTypes_u + detail::Num_PrimTypes_f,

    Num_Nodes = detail::Num_Nodes,

    Num_ArithOps = detail::Num_ArithOps,
    Num_RelOps   = detail::Num_RelOps,
    Num_ConvOps  = detail::Num_ConvOps,

    Num_Indexes = Num_Nodes + Num_ArithOps 
                            + Num_RelOps 
                            + Num_ConvOps
};

enum PrimTypeKind {
#define ANYDSL_U_TYPE(T) PrimType_##T,
#define ANYDSL_F_TYPE(T) PrimType_##T,
#include "anydsl/tables/typetable.h"
};

enum IndexKind {
#define ANYDSL_AIR_NODE(node) Index_##node,
#include "anydsl/tables/airnodetable.h"

#define ANYDSL_ARITHOP(op) Index_##op,
#include "anydsl/tables/arithoptable.h"

#define ANYDSL_RELOP(op) Index_##op,
#include "anydsl/tables/reloptable.h"

#define ANYDSL_CONVOP(op) Index_##op,
#include "anydsl/tables/convoptable.h"
};

enum NodeKind {
#define ANYDSL_AIR_NODE(node) Node_##node = Index_##node,
#include "anydsl/tables/airnodetable.h"
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
