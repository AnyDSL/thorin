#ifndef ANYDSL2_GLUE
#error "define ANYDSL2_GLUE before including this file"
#endif

#ifndef ANYDSL2_AIR_NODE
#error "define ANYDSL2_AIR_NODE before including this file"
#endif

#ifndef ANYDSL2_PRIMTYPE
#error "define ANYDSL2_PRIMTYPE before including this file"
#endif

#ifndef ANYDSL2_ARITHOP
#error "define ANYDSL2_ARITHOP before including this file"
#endif

#ifndef ANYDSL2_RELOP
#error "define ANYDSL2_RELOP before including this file"
#endif

#ifndef ANYDSL2_CONVOP
#error "define ANYDSL2_CONVOP before including this file"
#endif

#include "anydsl2/tables/nodetable.h"
    ANYDSL2_GLUE(Node, PrimType_u)
#define ANYDSL2_JUST_U_TYPE(T) ANYDSL2_PRIMTYPE(T)
#include "anydsl2/tables/primtypetable.h"
    ANYDSL2_GLUE(PrimType_u, PrimType_f)
#define ANYDSL2_JUST_F_TYPE(T) ANYDSL2_PRIMTYPE(T)
#include "anydsl2/tables/primtypetable.h"
    ANYDSL2_GLUE(PrimType_f, ArithOp)
#include "anydsl2/tables/arithoptable.h"
    ANYDSL2_GLUE(ArithOp, RelOp)
#include "anydsl2/tables/reloptable.h"
    ANYDSL2_GLUE(RelOp, ConvOp)
#include "anydsl2/tables/convoptable.h"

#undef ANYDSL2_GLUE
#undef ANYDSL2_AIR_NODE
#undef ANYDSL2_PRIMTYPE
#undef ANYDSL2_ARITHOP
#undef ANYDSL2_RELOP
#undef ANYDSL2_CONVOP
