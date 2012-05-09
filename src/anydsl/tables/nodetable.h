#ifndef ANYDSL_AIR_NODE
#error "define ANYDSL_AIR_NODE before including this file"
#endif

ANYDSL_AIR_NODE(Use)
    // Def
        // Type
            ANYDSL_AIR_NODE(PrimType)
            ANYDSL_AIR_NODE(Pi)
            ANYDSL_AIR_NODE(Sigma)
        // Literal
            ANYDSL_AIR_NODE(Prim)
            ANYDSL_AIR_NODE(Lambda)
            ANYDSL_AIR_NODE(Tuple)
        // Param
            ANYDSL_AIR_NODE(LParam)
            // PrimOp
                ANYDSL_AIR_NODE(ArithOp)
                ANYDSL_AIR_NODE(RelOp)
                ANYDSL_AIR_NODE(ConvOp)

#undef ANYDSL_AIR_NODE
