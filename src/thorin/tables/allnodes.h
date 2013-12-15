#ifndef THORIN_GLUE
#error "define THORIN_GLUE before including this file"
#endif

#ifndef THORIN_AIR_NODE
#error "define THORIN_AIR_NODE before including this file"
#endif

#ifndef THORIN_PRIMTYPE
#error "define THORIN_PRIMTYPE before including this file"
#endif

#ifndef THORIN_ARITHOP
#error "define THORIN_ARITHOP before including this file"
#endif

#ifndef THORIN_RELOP
#error "define THORIN_RELOP before including this file"
#endif

#ifndef THORIN_CONVOP
#error "define THORIN_CONVOP before including this file"
#endif

#include "thorin/tables/nodetable.h"
    THORIN_GLUE(Node, PrimType_u)
#define THORIN_U_TYPE(T) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_u, PrimType_f)
#define THORIN_F_TYPE(T) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_f, ArithOp)
#include "thorin/tables/arithoptable.h"
    THORIN_GLUE(ArithOp, RelOp)
#include "thorin/tables/reloptable.h"
    THORIN_GLUE(RelOp, ConvOp)
#include "thorin/tables/convoptable.h"

#undef THORIN_GLUE
#undef THORIN_AIR_NODE
#undef THORIN_PRIMTYPE
#undef THORIN_ARITHOP
#undef THORIN_RELOP
#undef THORIN_CONVOP
