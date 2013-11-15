#ifndef THORIN_CONVOP
#error "define THORIN_CONVOP before including this file"
#endif

THORIN_CONVOP(trunc)
THORIN_CONVOP(zext)
THORIN_CONVOP(sext)
THORIN_CONVOP(stof)
THORIN_CONVOP(utof)
THORIN_CONVOP(ftrunc)
THORIN_CONVOP(fext)
THORIN_CONVOP(ftos)
THORIN_CONVOP(ftou)
THORIN_CONVOP(bitcast)
THORIN_CONVOP(inttoptr)
THORIN_CONVOP(ptrtoint)

#undef THORIN_CONVOP
