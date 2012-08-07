#ifndef ANYDSL_CONVOP
#error "define ANYDSL_CONVOP before including this file"
#endif

ANYDSL_CONVOP(trunc)
ANYDSL_CONVOP(zext)
ANYDSL_CONVOP(sext)
ANYDSL_CONVOP(stof)
ANYDSL_CONVOP(utof)
ANYDSL_CONVOP(ftrunc)
ANYDSL_CONVOP(fext)
ANYDSL_CONVOP(ftos)
ANYDSL_CONVOP(ftou)
ANYDSL_CONVOP(bitcast)

#undef ANYDSL_CONVOP
