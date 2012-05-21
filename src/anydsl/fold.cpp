#include "anydsl/fold.h"

namespace anydsl {

FoldRes fold_bin(IndexKind kind, PrimTypeKind type, Box a, Box b) {
    FoldRes res;

    if (isArithOp(kind)) {
        res.value = fold_arith((ArithOpKind) kind, type, a, b, res.error);
        res.type = type;
        return res;
    } else if (isRelOp(kind)) {
        res.value = fold_rel((RelOpKind) kind, type, a, b, res.error);
        res.type = PrimType_u1;
        return res;
    } 

    ANYDSL_UNREACHABLE;
}

Box fold_arith(ArithOpKind kind, PrimTypeKind type, Box a, Box b, bool& error) {
    error = false;

    switch (kind) {
        case ArithOp_add:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(T(a.get_##T() + b.get_##T()));
#define ANYDSL_F_TYPE(T) case PrimType_##T: return Box(T(a.get_##T() + b.get_##T()));
#include "anydsl/tables/primtypetable.h"
            }
        case ArithOp_sub:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(T(a.get_##T() - b.get_##T()));
#define ANYDSL_F_TYPE(T) case PrimType_##T: return Box(T(a.get_##T() - b.get_##T()));
#include "anydsl/tables/primtypetable.h"
            }
        case ArithOp_mul:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(T(a.get_##T() * b.get_##T()));
#define ANYDSL_F_TYPE(T) case PrimType_##T: return Box(T(a.get_##T() * b.get_##T()));
#include "anydsl/tables/primtypetable.h"
            }
        default: ANYDSL_NOT_IMPLEMENTED;
    }
}

Box fold_rel(RelOpKind kind, PrimTypeKind type, Box a, Box b, bool& error) {
    error = false;

    switch (kind) {
        case RelOp_cmp_eq:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(a.get_##T() == b.get_##T());
#include "anydsl/tables/primtypetable.h"
                case PrimType_f32:
                case PrimType_f64: ANYDSL_UNREACHABLE;
            }
        case RelOp_cmp_ne:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(a.get_##T() != b.get_##T());
#include "anydsl/tables/primtypetable.h"
                case PrimType_f32:
                case PrimType_f64: ANYDSL_UNREACHABLE;
            }
        case RelOp_cmp_ult:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(a.get_##T() <  b.get_##T());
#include "anydsl/tables/primtypetable.h"
                case PrimType_f32:
                case PrimType_f64: ANYDSL_UNREACHABLE;
            }
        case RelOp_cmp_ule:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(a.get_##T() <= b.get_##T());
#include "anydsl/tables/primtypetable.h"
                case PrimType_f32:
                case PrimType_f64: ANYDSL_UNREACHABLE;
            }
        case RelOp_cmp_ugt:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(a.get_##T() >  b.get_##T());
#include "anydsl/tables/primtypetable.h"
                case PrimType_f32:
                case PrimType_f64: ANYDSL_UNREACHABLE;
            }
        case RelOp_cmp_uge:
            switch (type) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return Box(a.get_##T() >= b.get_##T());
#include "anydsl/tables/primtypetable.h"
                case PrimType_f32:
                case PrimType_f64: ANYDSL_UNREACHABLE;
            }

        default: ANYDSL_NOT_IMPLEMENTED;
    }
}

} // namespace anydsl
