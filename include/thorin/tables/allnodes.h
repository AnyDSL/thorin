#ifndef THORIN_GLUE
#error "define THORIN_GLUE before including this file"
#endif

#ifndef THORIN_PRIMTYPE
#error "define THORIN_PRIMTYPE before including this file"
#endif

#include "thorin/tables/nodetable.h"
    THORIN_GLUE(Node, PrimType_bool)
#define THORIN_BOOL_TYPE(T, M) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_bool, PrimType_ps)
#define THORIN_PS_TYPE(T, M) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_ps, PrimType_pu)
#define THORIN_PU_TYPE(T, M) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_pu, PrimType_qs)
#define THORIN_QS_TYPE(T, M) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_qs, PrimType_qu)
#define THORIN_QU_TYPE(T, M) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_qu, PrimType_pf)
#define THORIN_PF_TYPE(T, M) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_pf, PrimType_qf)
#define THORIN_QF_TYPE(T, M) THORIN_PRIMTYPE(T)
#include "thorin/tables/primtypetable.h"
    THORIN_GLUE(PrimType_qf, ArithOp)
#include "thorin/tables/arithoptable.h"
    THORIN_GLUE(ArithOp, Cmp)
#include "thorin/tables/cmptable.h"
    THORIN_GLUE(Cmp, MathOp)
#include "thorin/tables/mathoptable.h"

#undef THORIN_GLUE
#undef THORIN_PRIMTYPE
