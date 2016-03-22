#ifndef THORIN_AIR_NODE
#error "define THORIN_AIR_NODE before including this file"
#endif

// Def
    THORIN_AIR_NODE(Continuation, continuation)
    // PrimOp
        // Literal
            THORIN_AIR_NODE(Bottom, bottom)
            THORIN_AIR_NODE(MemBlob, mem_blob)
            THORIN_AIR_NODE(BlobPtr, mem_blob)
        // MemOp
            THORIN_AIR_NODE(Alloc, alloc)
            // Access
                THORIN_AIR_NODE(Load, load)
                THORIN_AIR_NODE(Store, store)
            THORIN_AIR_NODE(Enter, enter)
            THORIN_AIR_NODE(Leave, leave)
        THORIN_AIR_NODE(Select, select)
        THORIN_AIR_NODE(Global, global)
        THORIN_AIR_NODE(Slot, slot)
        //THORIN_AIR_NODE(ArithOp)
        //THORIN_AIR_NODE(Cmp)
        //THORIN_AIR_NODE(Conv, conv)
            THORIN_AIR_NODE(Cast, cast)
            THORIN_AIR_NODE(Bitcast, bitcast)
        THORIN_AIR_NODE(DefiniteArray, definite_array)
        THORIN_AIR_NODE(IndefiniteArray, indefinite_array)
        THORIN_AIR_NODE(Tuple, tuple)
        THORIN_AIR_NODE(StructAgg, struct_agg)
        THORIN_AIR_NODE(Vector, vector)
        THORIN_AIR_NODE(Extract, extract)
        THORIN_AIR_NODE(Insert, insert)
        THORIN_AIR_NODE(LEA, lea)
        THORIN_AIR_NODE(Run, run)
        THORIN_AIR_NODE(Hlt, hlt)
        THORIN_AIR_NODE(EndRun, end_run)
        THORIN_AIR_NODE(EndHlt, end_hlt)
    THORIN_AIR_NODE(Param, param)
    // Type
        // PrimType
        THORIN_AIR_NODE(TypeAbs, type_abs)
        THORIN_AIR_NODE(FrameType, frame)
        THORIN_AIR_NODE(TypeParam, type_param)
        THORIN_AIR_NODE(MemType, mem)
        THORIN_AIR_NODE(FnType, fn)
        THORIN_AIR_NODE(PtrType, ptr)
        THORIN_AIR_NODE(StructAbsType, struct_abs_type)
        THORIN_AIR_NODE(StructAppType, struct_app_type)
        THORIN_AIR_NODE(TupleType, tuple_type)
        THORIN_AIR_NODE(DefiniteArrayType, definite_array_type)
        THORIN_AIR_NODE(IndefiniteArrayType, indefinite_array_type)

#undef THORIN_AIR_NODE
