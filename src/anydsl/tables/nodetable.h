#ifndef ANYDSL_AIR_NODE
#error "define ANYDSL_AIR_NODE before including this file"
#endif

ANYDSL_AIR_NODE(Goto)
// Def
    ANYDSL_AIR_NODE(Lambda)
    // Literal
        // PrimLit
        ANYDSL_AIR_NODE(Undef)
        ANYDSL_AIR_NODE(ErrorLit)
    // Value
        // PrimOp
            //ANYDSL_AIR_NODE(ArithOp)
            //ANYDSL_AIR_NODE(RelOp)
            //ANYDSL_AIR_NODE(ConvOp)
            ANYDSL_AIR_NODE(Extract)
            ANYDSL_AIR_NODE(Insert)
            ANYDSL_AIR_NODE(Tuple)
            ANYDSL_AIR_NODE(Select)
        ANYDSL_AIR_NODE(Param)
// Type
    ANYDSL_AIR_NODE(NoRet)
    // PrimType
    ANYDSL_AIR_NODE(Pi)
    ANYDSL_AIR_NODE(Sigma)

#undef ANYDSL_AIR_NODE
