#ifndef ANYDSL_GLUE
#error "define ANYDSL_GLUE before including this file"
#endif

#ifndef ANYDSL_AIR_NODE
#error "define ANYDSL_AIR_NODE before including this file"
#endif

#ifndef ANYDSL_PRIMTYPE
#error "define ANYDSL_PRIMTYPE before including this file"
#endif

#ifndef ANYDSL_PRIMLIT
#error "define ANYDSL_PRIMLIT before including this file"
#endif

#ifndef ANYDSL_ARITHOP
#error "define ANYDSL_ARITHOP before including this file"
#endif

#ifndef ANYDSL_RELOP
#error "define ANYDSL_RELOP before including this file"
#endif

#ifndef ANYDSL_CONVOP
#error "define ANYDSL_CONVOP before including this file"
#endif

#include "anydsl/tables/nodetable.h"
    ANYDSL_GLUE(Node, PrimType_u)
#define ANYDSL_U_TYPE(T) ANYDSL_PRIMTYPE(T)
#include "anydsl/tables/primtypetable.h"
    ANYDSL_GLUE(PrimType_u, PrimType_f)
#define ANYDSL_F_TYPE(T) ANYDSL_PRIMTYPE(T)
#include "anydsl/tables/primtypetable.h"
    ANYDSL_GLUE(PrimType_f, PrimLit_u)
#define ANYDSL_U_TYPE(T) ANYDSL_PRIMLIT(T)
#include "anydsl/tables/primtypetable.h"
    ANYDSL_GLUE(PrimLit_u, PrimLit_f)
#define ANYDSL_F_TYPE(T) ANYDSL_PRIMLIT(T)
#include "anydsl/tables/primtypetable.h"
    ANYDSL_GLUE(PrimLit_f, ArithOp)
#include "anydsl/tables/arithoptable.h"
    ANYDSL_GLUE(ArithOp, RelOp)
#include "anydsl/tables/reloptable.h"
    ANYDSL_GLUE(RelOp, ConvOp)
#include "anydsl/tables/convoptable.h"

#undef ANYDSL_GLUE
#undef ANYDSL_AIR_NODE
#undef ANYDSL_PRIMTYPE
#undef ANYDSL_PRIMLIT
#undef ANYDSL_ARITHOP
#undef ANYDSL_RELOP
#undef ANYDSL_CONVOP
