#ifndef ANYDSL2_AIR_NODE
#error "define ANYDSL2_AIR_NODE before including this file"
#endif

// Def
    ANYDSL2_AIR_NODE(Lambda, lambda)
    // PrimOp
        // Literal
            ANYDSL2_AIR_NODE(PrimLit, primlit)
            ANYDSL2_AIR_NODE(Any, any)
            ANYDSL2_AIR_NODE(Bottom, bottom)
            ANYDSL2_AIR_NODE(TypeKeeper, keep)
        // MemOp
            ANYDSL2_AIR_NODE(Load, load)
            ANYDSL2_AIR_NODE(Store, store)
            ANYDSL2_AIR_NODE(Enter, enter)
            ANYDSL2_AIR_NODE(Leave, leave)
            ANYDSL2_AIR_NODE(CCall, call)
        ANYDSL2_AIR_NODE(LEA, lea)
        ANYDSL2_AIR_NODE(Slot, slot)
        //ANYDSL2_AIR_NODE(ArithOp)
        //ANYDSL2_AIR_NODE(RelOp)
        //ANYDSL2_AIR_NODE(ConvOp)
        ANYDSL2_AIR_NODE(Tuple, tuple)
        ANYDSL2_AIR_NODE(TupleExtract, tuple_extract)
        ANYDSL2_AIR_NODE(TupleInsert, tuple_insert)
        ANYDSL2_AIR_NODE(Select, select)
        ANYDSL2_AIR_NODE(Vector, vector)
    ANYDSL2_AIR_NODE(Param, param)
    // Type
        // PrimType
        ANYDSL2_AIR_NODE(Frame, frame)
        ANYDSL2_AIR_NODE(Generic, generic)
        ANYDSL2_AIR_NODE(Mem, mem)
        ANYDSL2_AIR_NODE(Opaque, opaque)
        ANYDSL2_AIR_NODE(Pi, pi)
        ANYDSL2_AIR_NODE(Ptr, ptr)
        ANYDSL2_AIR_NODE(Sigma, sigma)

#undef ANYDSL2_AIR_NODE
