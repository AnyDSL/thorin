#include "anydsl/fold.h"

namespace anydsl {

#define ANYDSL_NO_U_TYPE \
    case PrimType_u1: \
    case PrimType_u8: \
    case PrimType_u16: \
    case PrimType_u32: \
    case PrimType_u64: ANYDSL_UNREACHABLE;

#define ANYDSL_NO_F_TYPE \
    case PrimType_f32: \
    case PrimType_f64: ANYDSL_UNREACHABLE;

FoldValue fold_bin(IndexKind kind, PrimTypeKind type, FoldValue vl, FoldValue vr) {
    FoldValue res(res.type = isRelOp(kind) ? PrimType_u1 : type);

    // if one of the operands is Error -> return Error
    if (vl.kind == FoldValue::Error || vr.kind == FoldValue::Error) {
        res.kind = FoldValue::Error;
        return res;
    }

    // if one of the operands is Undef vlrious things may happen
    if (vl.kind == FoldValue::Undef || vr.kind == FoldValue::Undef) {
        switch (kind) {
            case Index_udiv:
            case Index_sdiv:
            case Index_fdiv:
                if (vr.kind == FoldValue::Undef) {
                    // assume division by zero
                    res.kind = FoldValue::Error;
                    return res;
                }
                // else fall through to default case
            case Index_bit_and: {
                // assume 000...0
                switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(0)); return res;
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            }
            case Index_bit_or: {
                // assume 111...1
                switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(-1)); return res;
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            }
            default:
                res.kind = FoldValue::Undef;
                return res;
        }
    }
                    
    /*
     * try to find identities if at least one vllue is Const
     */

    bool lconst = vl.kind == FoldValue::Const;
    bool rconst = vr.kind == FoldValue::Const;

    Box& l = vl.box;
    Box& r = vr.box;

    if (lconst || rconst) {
        // optimistically assume that we find an identity
        res.kind = FoldValue::Const;
        Box& c = lconst ? l : r;

        switch (kind) {
            case Index_add:
                switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: if (c.get_##T() == 0) { res.box = Box(T(0)); return res; }
#define ANYDSL_F_TYPE(T) case PrimType_##T: if (c.get_##T() == 0) { res.box = Box(T(0)); return res; }
#include "anydsl/tables/primtypetable.h"
                }
            default: { /* fall through */ }
        }
    }

    /*
     * if we didn't find any identities and at least one value is valid
     * we have to return an unknown valid result.
     */

    if (vl.kind == FoldValue::Valid || vr.kind == FoldValue::Valid) {
        res.kind = FoldValue::Valid;
        return res;
    }

    /*
     * Error and Undef cases have already been handled.
     * From now on we know that both a and b are Const.
     * However, the operations itself may still produces Error or Undef values.
     */

    anydsl_assert(vl.kind == FoldValue::Const && vr.kind == FoldValue::Const, "must both be constants");
    res.kind = FoldValue::Const;

    switch (kind) {

        /*
         * ArithOps
         */

        case Index_add:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() + r.get_##T())); return res;
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() + r.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
            }
        case ArithOp_sub:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() - r.get_##T())); return res;
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() - r.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
            }
        case Index_mul:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() * r.get_##T())); return res;
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() * r.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
            }
        case Index_udiv:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() / r.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_sdiv:
            switch (type) {
#define ANYDSL_U_TYPE(T) \
                case PrimType_##T: { \
                    typedef make_signed<T>::type S; \
                    res.box = Box(bcast<T , S>(bcast<S, T >(l.get_##T()) / bcast<S, T >(r.get_##T()))); \
                    return res; \
                }
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_fdiv:
            switch (type) {
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(l.get_##T() / r.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_U_TYPE;
            }

        /*
         * RelOps
         */

        case Index_cmp_eq:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(l.get_##T() == r.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ne:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(l.get_##T() != r.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ult:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(l.get_##T() <  r.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ule:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(l.get_##T() <= r.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ugt:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(l.get_##T() >  r.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_uge:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(l.get_##T() >= r.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }

        default: ANYDSL_NOT_IMPLEMENTED;
    }
}

} // namespace anydsl
