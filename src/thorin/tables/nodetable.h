#ifndef THORIN_NODE
#error "define THORIN_NODE before including this file"
#endif

// Def
    THORIN_NODE(Continuation, continuation)
    // PrimOp
        // Literal
            THORIN_NODE(Top, top)
            THORIN_NODE(Bottom, bottom)
            THORIN_NODE(MemBlob, mem_blob)
            THORIN_NODE(BlobPtr, mem_blob)
        // MemOp
            THORIN_NODE(Alloc, alloc)
            // Access
                THORIN_NODE(Load, load)
                THORIN_NODE(Store, store)
            THORIN_NODE(Enter, enter)
            THORIN_NODE(Leave, leave)
        THORIN_NODE(Select, select)
        THORIN_NODE(AlignOf, align_of)
        THORIN_NODE(SizeOf, size_of)
        THORIN_NODE(Global, global)
        THORIN_NODE(Slot, slot)
        //THORIN_NODE(ArithOp)
        //THORIN_NODE(Cmp)
        //THORIN_NODE(Conv, conv)
            THORIN_NODE(Cast, cast)
            THORIN_NODE(Bitcast, bitcast)
        THORIN_NODE(DefiniteArray, definite_array)
        THORIN_NODE(IndefiniteArray, indefinite_array)
        THORIN_NODE(Tuple, tuple)
        THORIN_NODE(Variant, variant)
        THORIN_NODE(VariantIndex, variant_index)
        THORIN_NODE(VariantExtract, variant_extract)
        THORIN_NODE(StructAgg, struct_agg)
        THORIN_NODE(Vector, vector)
        THORIN_NODE(Closure, closure)
        THORIN_NODE(Extract, extract)
        THORIN_NODE(Insert, insert)
        THORIN_NODE(LEA, lea)
        THORIN_NODE(Hlt, hlt)
        THORIN_NODE(Known, known)
        THORIN_NODE(Run, run)
        THORIN_NODE(Assembly, asm)
    THORIN_NODE(Param, param)
    THORIN_NODE(Filter, filter)
    // Type
        // PrimType
        THORIN_NODE(Star, star)
        THORIN_NODE(App, app)
        THORIN_NODE(DefiniteArrayType, definite_array_type)
        THORIN_NODE(FnType, fn)
        THORIN_NODE(ClosureType, closure_type)
        THORIN_NODE(FrameType, frame)
        THORIN_NODE(IndefiniteArrayType, indefinite_array_type)
        THORIN_NODE(Lambda, lambda)
        THORIN_NODE(MemType, mem)
        THORIN_NODE(BotType, bot_ty)
        THORIN_NODE(ExternType, ext_ty)
        THORIN_NODE(PtrType, ptr)
        THORIN_NODE(StructType, struct_type)
        THORIN_NODE(VariantType, variant_type)
        THORIN_NODE(TupleType, tuple_type)
        THORIN_NODE(Var, var)

#undef THORIN_NODE
