#ifndef ANYDSL2_AIR_NODE
#error "define ANYDSL2_AIR_NODE before including this file"
#endif

// Def
    ANYDSL2_AIR_NODE(Lambda)
    // PrimOp
        // Literal
            ANYDSL2_AIR_NODE(PrimLit)
            ANYDSL2_AIR_NODE(Any)
            ANYDSL2_AIR_NODE(Bottom)
            ANYDSL2_AIR_NODE(TypeKeeper)
        // MemOp
            ANYDSL2_AIR_NODE(Load)
            ANYDSL2_AIR_NODE(Store)
            ANYDSL2_AIR_NODE(Enter)
            ANYDSL2_AIR_NODE(Leave)
            ANYDSL2_AIR_NODE(CCall)
        ANYDSL2_AIR_NODE(Slot)
        //ANYDSL2_AIR_NODE(ArithOp)
        //ANYDSL2_AIR_NODE(RelOp)
        //ANYDSL2_AIR_NODE(ConvOp)
        ANYDSL2_AIR_NODE(Extract)
        ANYDSL2_AIR_NODE(Insert)
        ANYDSL2_AIR_NODE(Tuple)
        ANYDSL2_AIR_NODE(Select)
    ANYDSL2_AIR_NODE(Param)
    // Type
        // PrimType
        ANYDSL2_AIR_NODE(Frame)
        ANYDSL2_AIR_NODE(Generic)
        ANYDSL2_AIR_NODE(Mem)
        ANYDSL2_AIR_NODE(Opaque)
        ANYDSL2_AIR_NODE(Pi)
        ANYDSL2_AIR_NODE(Ptr)
        ANYDSL2_AIR_NODE(Sigma)

#undef ANYDSL2_AIR_NODE
