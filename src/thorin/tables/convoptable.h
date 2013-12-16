#ifndef THORIN_CONVOP
#error "define THORIN_CONVOP before including this file"
#endif

THORIN_CONVOP(convert)
THORIN_CONVOP(bitcast)
THORIN_CONVOP(inttoptr)
THORIN_CONVOP(ptrtoint)

#undef THORIN_CONVOP
