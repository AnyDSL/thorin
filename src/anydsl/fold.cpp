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

FoldValue fold_bin(IndexKind kind, PrimTypeKind type, FoldValue va, FoldValue vb) {
    FoldValue res(res.type = isRelOp(kind) ? PrimType_u1 : type);

    // if one of the operands is Error -> return Error
    if (va.kind == FoldValue::Error || vb.kind == FoldValue::Error) {
        res.kind = FoldValue::Error;
        return res;
    }

    res.kind = FoldValue::Valid;

    // if one of the operands is Undef various things may happen
    if (va.kind == FoldValue::Undef || vb.kind == FoldValue::Undef) {
        switch (kind) {
            case Index_udiv:
            case Index_sdiv:
            case Index_fdiv:
                if (vb.kind == FoldValue::Undef) {
                    res.kind = FoldValue::Error;
                    return res; // due to division by zero
                }
                // else fall through to default case
            case Index_bit_and: {
                switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(0)); return res;
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            }
            case Index_bit_or: {
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
                    
    Box& a = va.box;
    Box& b = vb.box;

    switch (kind) {

        /*
         * ArithOps
         */

        case Index_add:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() + b.get_##T())); return res;
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() + b.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
            }
        case ArithOp_sub:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() - b.get_##T())); return res;
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() - b.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
            }
        case Index_mul:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() * b.get_##T())); return res;
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() * b.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
            }
        case Index_udiv:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() / b.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_sdiv:
            switch (type) {
#define ANYDSL_U_TYPE(T) \
                case PrimType_##T: { \
                    typedef make_signed<T>::type S; \
                    res.box = Box(bcast<T , S>(bcast<S, T >(a.get_##T()) / bcast<S, T >(b.get_##T()))); \
                    return res; \
                }
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_fdiv:
            switch (type) {
#define ANYDSL_F_TYPE(T) case PrimType_##T: res.box = Box(T(a.get_##T() / b.get_##T())); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_U_TYPE;
            }

        /*
         * RelOps
         */

        case Index_cmp_eq:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(a.get_##T() == b.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ne:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(a.get_##T() != b.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ult:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(a.get_##T() <  b.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ule:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(a.get_##T() <= b.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_ugt:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(a.get_##T() >  b.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }
        case Index_cmp_uge:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: res.box = Box(a.get_##T() >= b.get_##T()); return res;
#include "anydsl/tables/primtypetable.h"
                ANYDSL_NO_F_TYPE;
            }

        default: ANYDSL_NOT_IMPLEMENTED;
    }
}

} // namespace anydsl
