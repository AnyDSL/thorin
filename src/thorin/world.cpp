#include "thorin/world.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>

#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/memop.h"
#include "thorin/type.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/clone_bodies.h"
#include "thorin/transform/lift_builtins.h"
#include "thorin/transform/lower2cff.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/mem2reg.h"
#include "thorin/transform/merge_lambdas.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/util/array.h"
#include "thorin/be/thorin.h"

#define THORIN_NO_U_TYPE \
    case PrimType_u1: \
    case PrimType_u8: \
    case PrimType_u16: \
    case PrimType_u32: \
    case PrimType_u64: THORIN_UNREACHABLE;

#define THORIN_NO_F_TYPE \
    case PrimType_f32: \
    case PrimType_f64: THORIN_UNREACHABLE;

#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
#define THORIN_BREAK asm("int3");
#else
#define THORIN_BREAK { int* __p__ = nullptr; *__p__ = 42; }
#endif

#ifndef NDEBUG
#define THORIN_CHECK_BREAK(gid) \
    if (breakpoints_.find((gid)) != breakpoints_.end()) { THORIN_BREAK }
#else
#define THORIN_CHECK_BREAK(gid) {}
#endif

namespace thorin {

/*
 * constructor and destructor
 */

World::World(std::string name)
    : name_(name)
    , primops_(1031)
    , types_(1031)
    , gid_(0)
    , sigma0_ (keep(new Sigma(*this, ArrayRef<const Type*>())))
    , pi0_    (keep(new Pi   (*this, ArrayRef<const Type*>())))
    , mem_    (keep(new Mem  (*this)))
    , frame_  (keep(new Frame(*this)))
#define THORIN_ALL_TYPE(T) ,T##_(keep(new PrimType(*this, PrimType_##T, 1)))
#include "thorin/tables/primtypetable.h"
{}

World::~World() {
    for (auto primop : primops_) delete primop;
    for (auto type   : types_  ) delete type;
    for (auto lambda : lambdas_) delete lambda;
}

Array<Lambda*> World::copy_lambdas() const {
    Array<Lambda*> result(lambdas().size());
    std::copy(lambdas().begin(), lambdas().end(), result.begin());
    return result;
}

/*
 * types
 */

Sigma* World::named_sigma(size_t size, const std::string& name) {
    Sigma* s = new Sigma(*this, size, name);
    assert(types_.find(s) == types_.end() && "must not be inside");
    types_.insert(s);
    return s;
}

/*
 * literals
 */

Def World::literal(PrimTypeKind kind, int value, size_t length) {
    Def lit;
    switch (kind) {
#define THORIN_U_TYPE(T) case PrimType_##T: lit = literal(T(value), 1); break;
#define THORIN_F_TYPE(T) THORIN_U_TYPE(T)
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE;
    }

    return vector(lit, length);
}

Def World::literal(PrimTypeKind kind, Box box, size_t length) { return vector(cse(new PrimLit(*this, kind, box, "")), length); }
Def World::any    (const Type* type, size_t length) { return vector(cse(new Any(type, "")), length); }
Def World::bottom (const Type* type, size_t length) { return vector(cse(new Bottom(type, "")), length); }
Def World::zero   (const Type* type, size_t length) { return zero  (type->as<PrimType>()->primtype_kind(), length); }
Def World::one    (const Type* type, size_t length) { return one   (type->as<PrimType>()->primtype_kind(), length); }
Def World::allset (const Type* type, size_t length) { return allset(type->as<PrimType>()->primtype_kind(), length); }
const TypeKeeper* World::typekeeper(const Type* type, const std::string& name) { return cse(new TypeKeeper(type, name)); }

/*
 * create
 */

Def World::binop(int kind, Def cond, Def lhs, Def rhs, const std::string& name) {
    if (is_arithop(kind))
        return arithop((ArithOpKind) kind, cond, lhs, rhs);

    assert(is_relop(kind) && "must be a RelOp");
    return relop((RelOpKind) kind, cond, lhs, rhs);
}

Def World::arithop(ArithOpKind kind, Def cond, Def a, Def b, const std::string& name) {
    assert(a->type() == b->type());
    assert(a->type()->as<PrimType>()->length() == b->type()->as<PrimType>()->length());
    PrimTypeKind type = a->type()->as<PrimType>()->primtype_kind();

    // bottom op bottom -> bottom
    if (a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(type);

    auto llit = a->isa<PrimLit>();
    auto rlit = b->isa<PrimLit>();
    auto lvec = a->isa<Vector>();
    auto rvec = b->isa<Vector>();

    if (lvec && rvec) {
        auto cvec = cond->isa<Vector>();
        size_t num = lvec->type()->as<PrimType>()->length();
        Array<Def> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cvec && cvec->op(i)->is_zero() ? bottom(type, 1) :  arithop(kind, lvec->op(i), rvec->op(i));
        return vector(ops, name);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();

        switch (kind) {
            case ArithOp_add:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_sub:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_mul:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_udiv:
                switch (type) {
#define THORIN_U_TYPE(T) \
                    case PrimType_##T: \
                        return rlit->is_zero() \
                             ? bottom(type) \
                             : literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_sdiv:
                switch (type) {
#define THORIN_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return rlit->is_zero() \
                            ? bottom(type) \
                            : literal(type, Box((T) ((S) l.get_##T() / (S) r.get_##T()))); \
                    }
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_urem:
                switch (type) {
#define THORIN_U_TYPE(T) \
                    case PrimType_##T: \
                        return rlit->is_zero() \
                             ? bottom(type) \
                             : literal(type, Box(T(l.get_##T() % r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_srem:
                switch (type) {
#define THORIN_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return literal(type, Box((T) ((S) l.get_##T() % (S) r.get_##T()))); \
                    }
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_and:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_or:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_xor:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_shl:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() << r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_lshr:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() >> r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_ashr:
                switch (type) {
#define THORIN_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return literal(type, Box((T) ((S) l.get_##T() >> (S) r.get_##T()))); \
                    }
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case ArithOp_fadd:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ArithOp_fsub:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ArithOp_fmul:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ArithOp_fdiv:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ArithOp_frem:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(type, Box(std::fmod(l.get_##T(), r.get_##T())));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
        }
    }

    if (a == b) {
        switch (kind) {
            case ArithOp_add:  return arithop_mul(cond, literal(type, 2), a);

            case ArithOp_sub:
            case ArithOp_srem:
            case ArithOp_urem:
            case ArithOp_xor:  return zero(type);

            case ArithOp_sdiv:
            case ArithOp_udiv: return one(type);

            case ArithOp_and:
            case ArithOp_or:   return a;

            default: break;
        }
    }

    if (a->is_zero()) {
        switch (kind) {
            case ArithOp_sdiv:
            case ArithOp_udiv:
            case ArithOp_srem:
            case ArithOp_urem: return bottom(type);

            case ArithOp_shl:
            case ArithOp_ashr:
            case ArithOp_lshr: return a;

            default: break;
        }
    } else if (a->is_one()) {
        switch (kind) {
            case ArithOp_sdiv:
            case ArithOp_udiv: return a;

            case ArithOp_srem:
            case ArithOp_urem: return zero(type);

            default: break;
        }
    } else if (rlit && rlit->primlit_value<uint64_t>() >= uint64_t(num_bits(type))) {
        switch (kind) {
            case ArithOp_shl:
            case ArithOp_ashr:
            case ArithOp_lshr: return bottom(type);

            default: break;
        }
    }

    if (kind == ArithOp_xor && a->is_allset()) {    // is this a NOT
        if (b->is_not())                            // do we have ~~x?
            return b->as<ArithOp>()->rhs();
        if (auto relop = b->isa<RelOp>())   // do we have ~(a cmp b)?
            return this->relop(negate(relop->relop_kind()), cond, relop->lhs(), relop->rhs());
    }

    auto lrel = a->isa<RelOp>();
    auto rrel = b->isa<RelOp>();

    if (kind == ArithOp_or && lrel && rrel && lrel->lhs() == rrel->lhs() && lrel->rhs() == rrel->rhs() 
            && lrel->relop_kind() == negate(rrel->relop_kind()))
            return literal_u1(true);

    if (kind == ArithOp_and && lrel && rrel && lrel->lhs() == rrel->lhs() && lrel->rhs() == rrel->rhs() 
            && lrel->relop_kind() == negate(rrel->relop_kind()))
            return literal_u1(false);

    auto land = a->kind() == Node_and ? a->as<ArithOp>() : nullptr;
    auto rand = b->kind() == Node_and ? b->as<ArithOp>() : nullptr;

    // distributivity (a and b) or (a and c)
    if (kind == ArithOp_or && land && rand) {
        if (land->lhs() == rand->lhs())
            return arithop_and(cond, land->lhs(), arithop_or(cond, land->rhs(), rand->rhs()));
        if (land->rhs() == rand->rhs())
            return arithop_and(cond, land->rhs(), arithop_or(cond, land->lhs(), rand->lhs()));
    }

    auto lor = a->kind() == Node_or ? a->as<ArithOp>() : nullptr;
    auto ror = b->kind() == Node_or ? b->as<ArithOp>() : nullptr;

    // distributivity (a or b) and (a or c)
    if (kind == ArithOp_and && lor && ror) {
        if (lor->lhs() == ror->lhs())
            return arithop_or(cond, lor->lhs(), arithop_and(cond, lor->rhs(), ror->rhs()));
        if (lor->rhs() == ror->rhs())
            return arithop_or(cond, lor->rhs(), arithop_and(cond, lor->lhs(), ror->lhs()));
    }

    // absorption
    if (kind == ArithOp_and) {
        if (ror) {
            if (a == ror->lhs()) return ror->rhs();
            if (a == ror->rhs()) return ror->lhs();
        }
        if (lor) {
            if (a == lor->lhs()) return lor->rhs();
            if (a == lor->rhs()) return lor->lhs();
        }
    }

    // absorption
    if (kind == ArithOp_or) {
        if (rand) {
            if (a == rand->lhs()) return rand->rhs();
            if (a == rand->rhs()) return rand->lhs();
        }
        if (land) {
            if (a == land->lhs()) return land->rhs();
            if (a == land->rhs()) return land->lhs();
        }
    }

    if (kind == ArithOp_or) {
        if (lor && ror) {
            if (lor->lhs() == ror->lhs())
                return arithop_or(lor->rhs(), ror->rhs());
            if (lor->rhs() == ror->rhs())
                return arithop_or(lor->lhs(), ror->lhs());
        }
    }

    if (kind == ArithOp_and) {
        if (land && rand) {
            if (land->lhs() == rand->lhs())
                return arithop_and(land->rhs(), rand->rhs());
            if (land->rhs() == rand->rhs())
                return arithop_and(land->lhs(), rand->lhs());
        }
    }

    // normalize: a - b = a + -b
    if ((kind == ArithOp_sub || kind == ArithOp_fsub) && !a->is_minus_zero()) { 
        rlit = (b = arithop_minus(b))->isa<PrimLit>();
        kind = kind == ArithOp_sub ? ArithOp_add : ArithOp_fadd;
    }

    // normalize: swap literal/vector to the left
    if (is_commutative(kind) && (rlit || rvec)) {
        std::swap(a, b);
        std::swap(llit, rlit);
        std::swap(lvec, rvec);
    }

    if (a->is_zero()) {
        switch (kind) {
            // TODO: disable fast-math
            case ArithOp_fmul:

            case ArithOp_mul:
            case ArithOp_sdiv:
            case ArithOp_udiv:
            case ArithOp_srem:
            case ArithOp_urem:
            case ArithOp_and:
            case ArithOp_shl:
            case ArithOp_lshr:
            case ArithOp_ashr: return zero(type);

            // TODO: disable fast-math
            case ArithOp_fadd:

            case ArithOp_add: 
            case ArithOp_or:
            case ArithOp_xor:  return b;

            default: break;
        }
    } 
    if (a->is_one()) {
        switch (kind) {
            case ArithOp_mul: return b;
            default: break;
        }
    } 
    if (a->is_allset()) {
        switch (kind) {
            case ArithOp_and: return b;
            case ArithOp_or:  return llit; // allset
            default: break;
        }
    }

    // normalize: try to reorder same ops to have the literal/vector on the left-most side
    if (is_associative(kind)) {
        auto a_same = a->isa<ArithOp>() && a->as<ArithOp>()->arithop_kind() == kind ? a->as<ArithOp>() : nullptr;
        auto b_same = b->isa<ArithOp>() && b->as<ArithOp>()->arithop_kind() == kind ? b->as<ArithOp>() : nullptr;
        auto a_lhs_lv = a_same && (a_same->lhs()->isa<PrimLit>() || a_same->lhs()->isa<Vector>()) ? a_same->lhs() : nullptr;
        auto b_lhs_lv = b_same && (b_same->lhs()->isa<PrimLit>() || b_same->lhs()->isa<Vector>()) ? b_same->lhs() : nullptr;

        if (is_commutative(kind)) {
            if (a_lhs_lv && b_lhs_lv)
                return arithop(kind, arithop(kind, a_lhs_lv, b_lhs_lv), arithop(kind, a_same->rhs(), b_same->rhs()));
            if ((llit || lvec) && b_lhs_lv)
                return arithop(kind, arithop(kind, a, b_lhs_lv), b_same->rhs());
            if (b_lhs_lv)
                return arithop(kind, b_lhs_lv, arithop(kind, a, b_same->rhs()));
        }
        if (a_lhs_lv)
            return arithop(kind, a_lhs_lv, arithop(kind, a_same->rhs(), b));
    }

    return cse(new ArithOp(kind, cond, a, b, name));
}

Def World::arithop_not(Def cond, Def def) { return arithop_xor(cond, allset(def->type(), def->length()), def); }

Def World::arithop_minus(Def cond, Def def) {
    switch (PrimTypeKind kind = def->type()->as<PrimType>()->primtype_kind()) {
        case PrimType_f32: return arithop_fsub(cond, literal_f32(-0.f), def);
        case PrimType_f64: return arithop_fsub(cond, literal_f64(-0.0), def);
        default:           return arithop_sub(cond, zero(kind), def);
    }
}

Def World::relop(RelOpKind kind, Def cond, Def a, Def b, const std::string& name) {
    if (a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(type_u1());

    RelOpKind oldkind = kind;
    switch (kind) {
        case RelOp_cmp_ugt:  kind = RelOp_cmp_ult; break;
        case RelOp_cmp_uge:  kind = RelOp_cmp_ule; break;
        case RelOp_cmp_sgt:  kind = RelOp_cmp_slt; break;
        case RelOp_cmp_sge:  kind = RelOp_cmp_sle; break;
        case RelOp_fcmp_ogt: kind = RelOp_fcmp_olt; break;
        case RelOp_fcmp_oge: kind = RelOp_fcmp_ole; break;
        case RelOp_fcmp_ugt: kind = RelOp_fcmp_ult; break;
        case RelOp_fcmp_uge: kind = RelOp_fcmp_ule; break;
        default: break;
    }

    if (oldkind != kind)
        std::swap(a, b);

    auto llit = a->isa<PrimLit>();
    auto rlit = b->isa<PrimLit>();
    auto  lvec = a->isa<Vector>();
    auto  rvec = b->isa<Vector>();

    if (lvec && rvec) {
        size_t num = lvec->type()->as<PrimType>()->length();
        Array<Def> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = relop(kind, lvec->op(i), rvec->op(i));
        return vector(ops, name);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();
        PrimTypeKind type = llit->primtype_kind();

        switch (kind) {
            case RelOp_cmp_eq:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case RelOp_cmp_ne:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case RelOp_cmp_ult:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case RelOp_cmp_ule:
                switch (type) {
#define THORIN_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case RelOp_cmp_slt:
                switch (type) {
#define THORIN_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return literal_u1((S) l.get_##T() < (S) r.get_##T()); \
                    }
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case RelOp_cmp_sle:
                switch (type) {
#define THORIN_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed< T >::type S; \
                        return literal_u1((S) l.get_##T() <= (S) r.get_##T()); \
                    }
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_F_TYPE;
                }
            case RelOp_fcmp_oeq:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case RelOp_fcmp_one:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case RelOp_fcmp_olt:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case RelOp_fcmp_ole:
                switch (type) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            default:
                THORIN_UNREACHABLE;
        }
    }

    if (a == b) {
        switch (kind) {
            case RelOp_cmp_ult:
            case RelOp_cmp_slt: 
            case RelOp_cmp_eq:  return zero(u1_);
            case RelOp_cmp_ule:
            case RelOp_cmp_sle:
            case RelOp_cmp_ne:  return one(u1_);
            default: break;
        }
    }

    return cse(new RelOp(kind, cond, a, b, name));
}

static i64 box2i64(PrimTypeKind kind, Box box) {
    switch (kind) {
#define THORIN_U_TYPE(T) case PrimType_##T: return (i64) (make_signed<T>::type) box.get_##T();
#include "thorin/tables/primtypetable.h"
        THORIN_NO_F_TYPE;
        default: THORIN_UNREACHABLE;
    }
}

Def World::convop(ConvOpKind kind, Def cond, Def from, const Type* to, const std::string& name) {
#define from_kind (from->type()->as<PrimType>()->primtype_kind())
#define   to_kind (  to        ->as<PrimType>()->primtype_kind())
#ifndef NDEBUG
    switch (kind) {
        case ConvOp_trunc:      assert(num_bits(from_kind) > num_bits(to_kind)); break;
        case ConvOp_sext:
        case ConvOp_zext:       assert(num_bits(from_kind) < num_bits(to_kind)); break;
        case ConvOp_stof:  
        case ConvOp_utof:       assert(  is_int(from_kind) && is_float(to_kind)); break;
        case ConvOp_ftos:       
        case ConvOp_ftou:       assert(is_float(from_kind) &&   is_int(to_kind)); break;
        case ConvOp_ftrunc:     assert(from_kind == PrimType_f64 && to_kind == PrimType_f32); break;
        case ConvOp_fext:       assert(from_kind == PrimType_f32 && to_kind == PrimType_f64); break;
        case ConvOp_inttoptr:   assert(is_int(from_kind) && to->isa<Ptr>()); break;
        case ConvOp_ptrtoint:   assert(from->type()->isa<Ptr>() && is_int(to_kind)); break;
        case ConvOp_bitcast:    /* TODO check */;
    }
#endif

    if (from->isa<Bottom>())
        return bottom(to);

    auto lit = from->isa<PrimLit>();
    auto vec = from->isa<Vector>();

    if (vec) {
        auto cvec = cond->isa<Vector>();
        size_t num = vec->length();
        Array<Def> ops(num);
        auto to_scalar = to->as<VectorType>()->scalarize();
        for (size_t i = 0; i != num; ++i)
            ops[i] = cvec && cvec->op(i)->is_zero() ? bottom(to_scalar, 1) :  convop(kind, vec->op(i), to_scalar);
        return vector(ops, name);
    }

    if (lit) {
        Box box = lit->value();

        switch (kind) {
            case ConvOp_trunc:
            case ConvOp_zext:   return literal(to_kind, box);
            case ConvOp_sext:   return literal(to_kind, Box((u64) box2i64(from_kind, box)));
            case ConvOp_utof:
                switch (to_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((T) box.get_u64()));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_stof:
                switch (to_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((T) box2i64(from_kind, box)));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_ftou:
                switch (from_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((u64) box.get_##T()));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_ftos:
                switch (from_kind) {
#define THORIN_F_TYPE(T) case PrimType_##T: return literal(to_kind, Box((u64) (i64) box.get_##T()));
#include "thorin/tables/primtypetable.h"
                    THORIN_NO_U_TYPE;
                }
            case ConvOp_ftrunc: return literal(PrimType_f32, Box((f32) box.get_f64()));
            case ConvOp_fext:   return literal(PrimType_f64, Box((f64) box.get_f32()));
            case ConvOp_bitcast:
            case ConvOp_inttoptr:
            case ConvOp_ptrtoint: /* FALLTROUGH */;
        }
    }

    return cse(new ConvOp(kind, cond, from, to, name));
}

Def World::extract(Def agg, Def index, const std::string& name) {
    if (agg->isa<Bottom>())
        return bottom(Extract::type(agg, index));

    if (auto aggregate = agg->isa<Aggregate>())
        if (auto lit = index->isa<PrimLit>())
            return aggregate->op_via_lit(lit);

    if (auto insert = agg->isa<Insert>()) {
        if (index == insert->index())
            return insert->value();
        else if (index->isa<PrimLit>()) {
            if (insert->index()->isa<PrimLit>())
                return extract(insert->agg(), index);
        }
    }

    return cse(new Extract(agg, index, name));
}

Def World::insert(Def agg, Def index, Def value, const std::string& name) {
    if (agg->isa<Bottom>() || value->isa<Bottom>())
        return bottom(agg->type());

    if (auto aggregate = agg->isa<Aggregate>()) {
        if (auto literal = index->isa<PrimLit>()) {
            Array<Def> args(agg->size());
            std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
            args[literal->primlit_value<u64>()] = value;
            return rebuild(aggregate, args);
        }
    }

    return cse(new Insert(agg, index, value, name));
}

Def World::extract(Def tuple, u32 index, const std::string& name) { return extract(tuple, literal_u32(index), name); }
Def World::insert(Def tuple, u32 index, Def value, const std::string& name) { 
    return insert(tuple, literal_u32(index), value, name); 
}

Def World::vector(Def arg, size_t length, const std::string& name) {
    if (length == 1) 
        return arg;

    Array<Def> args(length);
    std::fill(args.begin(), args.end(), arg);
    return vector(args, name);
}

Def World::select(Def cond, Def a, Def b, const std::string& name) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type());

    if (auto lit = cond->isa<PrimLit>())
        return lit->value().get_u1().get() ? a : b;

    if (cond->is_not()) {
        cond = cond->as<ArithOp>()->rhs();
        std::swap(a, b);
    }

    if (a == b)
        return a;

    return cse(new Select(cond, a, b, name));
}

const Enter* World::enter(Def mem, const std::string& name) {
    if (auto leave = mem->isa<Leave>())
        return leave->frame();

    return cse(new Enter(mem, name));
}

Def World::leave(Def mem, Def frame, const std::string& name) { 
    for (auto use : frame->uses()) {
        if (use->isa<Slot>())
            return cse(new Leave(mem, frame, name)); 
    }

    return mem;
}

Def World::load(Def mem, Def ptr, const std::string& name) { 
    if (auto store = mem->isa<Store>()) {
        if (store->ptr() == ptr) {
            return store->val();
        }
    }

    if (auto global = ptr->isa<Global>()) {
        if (!global->is_mutable())
            return global->init();
    }


    return cse(new Load(mem, ptr, name)); 
}

const Store* World::store(Def mem, Def ptr, Def value, const std::string& name) { 
    if (auto store = mem->isa<Store>()) {
        if (ptr == store->ptr())
            mem = store->mem();
    }

    return cse(new Store(mem, ptr, value, name)); 
}

const LEA* World::lea(Def ptr, Def index, const std::string& name) { return cse(new LEA(ptr, index, name)); }
const Global* World::global(Def init, bool is_mutable, const std::string& name) { return cse(new Global(init, is_mutable, name)); }
const Slot* World::slot(const Type* type, Def frame, size_t index, const std::string& name) {
    return cse(new Slot(type, frame, index, name));
}

const Global* World::global_immutable_string(const std::string& str, const std::string& name) {
    size_t size = str.size() + 1;

    Array<Def> str_array(size);
    for (size_t i = 0; i != size-1; ++i)
        str_array[i] = literal_u8(str[i]);
    str_array.back() = literal_u8('\0');

    return global(array(str_array), false, name);
}

Def World::run(Def def, const std::string& name) { 
    if (auto run  = def->isa<Run >()) return run;
    if (auto halt = def->isa<Halt>()) return halt;
    return cse(new Run(def, name)); 
}

Def World::halt(Def def, const std::string& name) { 
    if (auto halt = def->isa<Halt>()) return halt;
    if (auto run  = def->isa<Run >()) 
        def = run->def();
    return cse(new Halt(def, name)); 
}

Lambda* World::lambda(const Pi* pi, Lambda::Attribute attribute, const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    auto l = new Lambda(gid_++, pi, attribute, true, name);
    lambdas_.insert(l);

    size_t i = 0;
    for (auto elem : pi->elems())
        l->params_.push_back(param(elem, l, i++));

    return l;
}

Lambda* World::basicblock(const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    auto bb = new Lambda(gid_++, pi0(), Lambda::Attribute(0), false, name);
    lambdas_.insert(bb);
    return bb;
}

Def World::rebuild(World& to, const PrimOp* in, ArrayRef<Def> ops, const Type* type) {
    NodeKind kind = in->kind();
    const std::string& name = in->name;
    assert(&type->world() == &to);
#ifndef NDEBUG
    for (auto op : ops)
        assert(&op->world() == &to);
#endif

    if (ops.empty() && &in->world() == &to) return in;
    if (is_arithop (kind))  { assert(ops.size() == 3); return to.arithop((ArithOpKind) kind, ops[0], ops[1], ops[2], name); }
    if (is_relop   (kind))  { assert(ops.size() == 3); return to.relop(  (RelOpKind  ) kind, ops[0], ops[1], ops[2], name); }
    if (is_convop  (kind))  { assert(ops.size() == 2); return to.convop( (ConvOpKind ) kind, ops[0], ops[1],   type, name); }
    if (is_primtype(kind)) { 
        assert(ops.size() == 0); 
        auto primlit = in->as<PrimLit>();
        return to.literal(primlit->primtype_kind(), primlit->value()); 
    }

    switch (kind) {
        case Node_Any:     assert(ops.size() == 0); return to.any(type);
        case Node_Bottom:  assert(ops.size() == 0); return to.bottom(type);
        case Node_Enter:   assert(ops.size() == 1); return to.enter(  ops[0], name);
        case Node_Extract: assert(ops.size() == 2); return to.extract(ops[0], ops[1], name);
        case Node_Global:  assert(ops.size() == 1); return to.global( ops[0], in->as<Global>()->is_mutable(), name);
        case Node_Halt:    assert(ops.size() == 1); return to.halt(   ops[0], name);
        case Node_Insert:  assert(ops.size() == 3); return to.insert( ops[0], ops[1], ops[2], name);
        case Node_LEA:     assert(ops.size() == 2); return to.lea(ops[0], ops[1], name);
        case Node_Leave:   assert(ops.size() == 2); return to.leave(  ops[0], ops[1], name);
        case Node_Load:    assert(ops.size() == 2); return to.load(   ops[0], ops[1], name);
        case Node_Run:     assert(ops.size() == 1); return to.run(    ops[0], name);
        case Node_Select:  assert(ops.size() == 3); return to.select( ops[0], ops[1], ops[2], name);
        case Node_Store:   assert(ops.size() == 3); return to.store(  ops[0], ops[1], ops[2], name);
        case Node_Tuple:                            return to.tuple(ops, name);
        case Node_Vector:                           return to.vector(ops, name);
        case Node_ArrayAgg:                         
            return to.array(type->as<ArrayType>()->elem_type(), ops, type->isa<DefArray>(), name);
        case Node_Slot:    assert(ops.size() == 1); 
            return to.slot(type->as<Ptr>()->referenced_type(), ops[0], in->as<Slot>()->index(), name);
        default: THORIN_UNREACHABLE;
    }
}

const Type* World::rebuild(World& to, const Type* type, ArrayRef<const Type*> elems) {
    if (elems.empty() && &type->world() == &to) 
        return type;

    if (is_primtype(type->kind())) {
        assert(elems.size() == 0); 
        auto primtype = type->as<PrimType>();
        return to.type(primtype->primtype_kind(), primtype->length()); 
    }

    switch (type->kind()) {
        case Node_DefArray:   assert(elems.size() == 1); return to.def_array(elems.front(), type->as<DefArray>()->dim());
        case Node_Generic:    assert(elems.size() == 0); return to.generic(type->as<Generic>()->index());
        case Node_IndefArray: assert(elems.size() == 1); return to.indef_array(elems.front());
        case Node_Mem:        assert(elems.size() == 0); return to.mem();
        case Node_Frame:      assert(elems.size() == 0); return to.frame();
        case Node_Ptr:        assert(elems.size() == 1); return to.ptr(elems.front(), type->as<Ptr>()->length());
        case Node_Sigma:      return to.sigma(elems);
        case Node_Pi:         return to.pi(elems);
        case Node_GenericRef: {
            auto genref = type->as<GenericRef>();
            return to.generic_ref(genref->generic(), genref->lambda());
        }
        default: THORIN_UNREACHABLE;
    }
}

const Param* World::param(const Type* type, Lambda* lambda, size_t index, const std::string& name) {
    THORIN_CHECK_BREAK(gid_)
    return new Param(gid_++, type, lambda, index, name);
}

/*
 * cse + unify
 */

const Type* World::unify_base(const Type* type) {
    auto i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        return *i;
    }

    auto p = types_.insert(type);
    assert(type->gid_ == size_t(-1));
    type->gid_ = gid_++;
    assert(p.second && "hash/equal broken");
    return type;
}

const DefNode* World::cse_base(const PrimOp* primop) {
    auto i = primops_.find(primop);
    if (i != primops_.end()) {
        for (size_t x = 0, e = primop->size(); x != e; ++x)
            primop->unregister_use(x);

        delete primop;
        primop = *i;
    } else {
        primop->set_gid(gid_++);
        auto p = primops_.insert(primop);
        assert(p.second && "hash/equal broken");
    }

    THORIN_CHECK_BREAK(primop->gid())
    return primop;
}

/*
 * optimizations
 */

void World::cleanup() {
    eliminate_params();
    unreachable_code_elimination();
    dead_code_elimination();
    unused_type_elimination();
    debug_verify(*this);
}

void World::opt() {
    cleanup();
    //partial_evaluation(*this);
    lower2cff(*this);
    clone_bodies(*this);
    mem2reg(*this);
    lift_builtins(*this);
    inliner(*this);
    merge_lambdas(*this);
    cleanup();
}

void World::eliminate_params() {
    for (auto olambda : copy_lambdas()) { 
        if (olambda->empty())
            continue;

        olambda->clear();
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;
        size_t i = 0;
        for (auto param : olambda->params()) {
            if (param->is_proxy())
                proxy_idx.push_back(i++);
            else
                param_idx.push_back(i++);
        }

        if (proxy_idx.empty()) 
            continue;

        auto nlambda = lambda(pi(olambda->type()->elems().cut(proxy_idx)), olambda->attribute(), olambda->name);
        size_t j = 0;
        for (auto i : param_idx) {
            olambda->param(i)->replace(nlambda->param(j));
            nlambda->param(j++)->name = olambda->param(i)->name;
        }

        nlambda->jump(olambda->to(), olambda->args());
        olambda->destroy_body();

        for (auto use : olambda->uses()) {
            auto ulambda = use->as_lambda();
            assert(use.index() == 0 && "deleted param of lambda used as argument");
            ulambda->jump(nlambda, ulambda->args().cut(proxy_idx));
        }
    }
}

void World::unreachable_code_elimination() {
    LambdaSet set;

    for (auto lambda : lambdas())
        if (lambda->attribute().is(Lambda::Extern))
            uce_insert(set, lambda);

    for (auto lambda : lambdas()) {
        if (!set.contains(lambda))
            lambda->destroy_body();
    }
}

void World::uce_insert(LambdaSet& set, Lambda* lambda) {
    if (set.visit(lambda)) return;
    for (auto succ : lambda->succs())
        uce_insert(set, succ);
}

static void sanity_check(Def def) {
    if (auto param = def->isa<Param>())
        assert(!param->lambda()->empty());
    else if (auto lambda = def->isa_lambda())
        assert(lambda->attribute().is(Lambda::Extern | Lambda::Intrinsic) || !lambda->empty());
}

void World::dead_code_elimination() {
    const auto old_gid = gid_;
    Def2Def map;
    for (auto lambda : lambdas()) {
        for (size_t i = 0, e = lambda->ops().size(); i != e; ++i)
            lambda->update_op(i, dce_rebuild(map, old_gid, lambda->op(i)));
    }

    DefSet set;
    for (auto lambda : lambdas()) {
        for (size_t i = 0, e = lambda->ops().size(); i != e; ++i)
            dce_mark(set, lambda->op(i));
    }

    auto wipe_primop = [&] (const PrimOp* primop) { return !set.contains(primop) && !primop->isa<TypeKeeper>(); };
    auto wipe_lambda = [&] (Lambda* lambda) {
        return !lambda->attribute().is(Lambda::Extern) 
            && (   (!lambda->attribute().is(Lambda::Intrinsic) && lambda->empty()) 
                || (lambda->attribute().is(Lambda::Intrinsic) && lambda->num_uses() == 0));
    };

    for (auto primop : primops_) {
        if (wipe_primop(primop)) {
            for (size_t i = 0, e = primop->size(); i != e; ++i)
                primop->unregister_use(i);
            if (primop->is_proxy()) {
                auto num = primop->representative_->representatives_of_.erase(primop);
                assert(num == 1);
            }
#ifndef NDEBUG
        } else {
            for (auto op : primop->ops())
                sanity_check(op);
#endif
        }
    }

    for (auto lambda : lambdas()) {
        if (wipe_lambda(lambda)) {
            for (auto param : lambda->params()) {
                if (param->is_proxy()) {
                    auto num = param->representative_->representatives_of_.erase(param);
                    assert(num == 1);
                }
            }
#ifndef NDEBUG
        } else {
            for (auto op : lambda->ops())
                sanity_check(op);
#endif
        }
    }

    wipe_out(primops_, wipe_primop);
    wipe_out(lambdas_, wipe_lambda);
    verify_closedness(*this);
}

Def World::dce_rebuild(Def2Def& map, const size_t old_gid, Def def) {
    if (const DefNode* mapped = map.find(def))
        return mapped;
    if (def->gid() >= old_gid)
        return def;
    if (def->isa<Lambda>() || def->isa<Param>())
        return map[def] = def;

    auto oprimop = def->as<PrimOp>();
    Array<Def> ops(oprimop->size());
    for (size_t i = 0, e = oprimop->size(); i != e; ++i)
        ops[i] = dce_rebuild(map, old_gid, oprimop->op(i));

    return map[oprimop] = rebuild(oprimop, ops);
}

void World::dce_mark(DefSet& set, Def def) {
    if (set.visit(def) || def->isa<Lambda>() || def->isa<Param>())
        return;

    for (auto op : def->as<PrimOp>()->ops())
        dce_mark(set, op);
}

void World::unused_type_elimination() {
    std::unordered_set<const Type*> set;

    for (auto primop : primops())
        ute_insert(set, primop->type());

    for (auto lambda : lambdas()) {
        ute_insert(set, lambda->type());
        for (auto param : lambda->params())
            ute_insert(set, param->type());
    }

    wipe_out(types_, [=] (const Type* type) { return set.find(type) == set.end(); });
}

void World::ute_insert(std::unordered_set<const Type*>& set, const Type* type) {
    assert(types_.find(type) != types_.end() && "not in map");

    if (set.find(type) != set.end()) return;
    set.insert(type);

    for (auto elem : type->elems())
        ute_insert(set, elem);
}

template<class S, class W>
void World::wipe_out(S& set, W wipe) {
    for (auto i = set.begin(); i != set.end();) {
        auto j = i++;
        auto val = *j;
        if (wipe(val)) {
            set.erase(j);
            delete val;
        }
    }
}

} // namespace thorin
