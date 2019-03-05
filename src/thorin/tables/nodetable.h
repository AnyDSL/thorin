#ifndef THORIN_NODE
#error "define THORIN_NODE before including this file"
#endif

// Def
    THORIN_NODE(KindArity, *A)      // don't
    THORIN_NODE(KindMulti, *M)      // change
    THORIN_NODE(KindStar,  *)       // the
    THORIN_NODE(Universe, universe) // order
    THORIN_NODE(App, app)
    THORIN_NODE(Axiom, axiom)
    THORIN_NODE(Bot, bot)
    THORIN_NODE(Extract, extract)
    THORIN_NODE(Insert, insert)
    THORIN_NODE(Lam, lam)
    THORIN_NODE(Lit, lit)
    THORIN_NODE(Pack, pack)
    THORIN_NODE(Pi, pi)
    THORIN_NODE(Sigma, sigma)
    THORIN_NODE(Top, top)
    THORIN_NODE(Tuple, tuple)
    THORIN_NODE(Var, var)
    THORIN_NODE(Variadic, variadic)
    THORIN_NODE(VariantType, variant_type)
    // get rid of these ones
    THORIN_NODE(DefiniteArrayType, definite_array_type)
    THORIN_NODE(FrameType, frame)
    THORIN_NODE(IndefiniteArrayType, indefinite_array_type)
    THORIN_NODE(MemType, mem)
    THORIN_NODE(PtrType, ptr)
    // PrimOp
        // MemOp
            THORIN_NODE(Alloc, alloc)
            // Access
                THORIN_NODE(Load, load)
                THORIN_NODE(Store, store)
            THORIN_NODE(Enter, enter)
            THORIN_NODE(Leave, leave)
        THORIN_NODE(Select, select)
        THORIN_NODE(SizeOf, size_of)
        THORIN_NODE(Global, global)
        THORIN_NODE(Slot, slot)
        //THORIN_NODE(ArithOp)
        //THORIN_NODE(Cmp)
        //THORIN_NODE(Conv, conv)
            THORIN_NODE(Cast, cast)
            THORIN_NODE(Bitcast, bitcast)
        THORIN_NODE(Nat, nat)
        THORIN_NODE(DefiniteArray, definite_array)
        THORIN_NODE(IndefiniteArray, indefinite_array)
        THORIN_NODE(Variant, variant)
        THORIN_NODE(LEA, lea)
        THORIN_NODE(Hlt, hlt)
        THORIN_NODE(Known, known)
        THORIN_NODE(Run, run)
        THORIN_NODE(Assembly, asm)
    THORIN_NODE(Param, param)
    THORIN_NODE(Analyze, analyze)
    // Type
        // PrimType

#undef THORIN_NODE
