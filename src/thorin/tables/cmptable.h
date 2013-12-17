#ifndef THORIN_CMP
#error "define THORIN_CMP before including this file"
#endif

#ifdef  THORIN_CMPOP
#error "THORIN_CMPOP"
#endif

#ifdef  THORIN_RELOP
#error "THORIN_RELOP"
#endif

THORIN_CMP(eq)
THORIN_CMP(ne)
THORIN_CMP(gt)
THORIN_CMP(ge)
THORIN_CMP(lt)
THORIN_CMP(le)

#undef THORIN_CMP
