#include "anydsl2/world.h"

#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>
#include <boost/unordered_map.hpp>

#include "anydsl2/def.h"
#include "anydsl2/primop.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/type.h"
#include "anydsl2/analyses/verifier.h"
#include "anydsl2/transform/cfg_builder.h"
#include "anydsl2/transform/inliner.h"
#include "anydsl2/transform/mem2reg.h"
#include "anydsl2/transform/merge_lambdas.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"
#include "anydsl2/util/hash.h"

#define ANYDSL2_NO_U_TYPE \
    case PrimType_u1: \
    case PrimType_u8: \
    case PrimType_u16: \
    case PrimType_u32: \
    case PrimType_u64: ANYDSL2_UNREACHABLE;

#define ANYDSL2_NO_F_TYPE \
    case PrimType_f32: \
    case PrimType_f64: ANYDSL2_UNREACHABLE;

#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
#define ANYDSL2_BREAK asm("int3");
#else
#define ANYDSL2_BREAK { int* __p__ = 0; *__p__ = 42; }
#endif

#ifndef NDEBUG
#define ANYDSL2_CHECK_BREAK \
    if (breakpoints_.find(gid_) != breakpoints_.end()) \
        ANYDSL2_BREAK;
#else
#define ANYDSL2_CHECK_BREAK {}
#endif

namespace anydsl2 {

/*
 * constructor and destructor
 */

World::World()
    : primops_(1031)
    , types_(1031)
    , gid_(0)
    , pass_counter_(1)
    , sigma0_ (keep(new Sigma(*this, ArrayRef<const Type*>())))
    , pi0_    (keep(new Pi   (*this, ArrayRef<const Type*>())))
    , mem_    (keep(new Mem  (*this)))
    , frame_  (keep(new Frame(*this)))
#define ANYDSL2_UF_TYPE(T) ,T##_(keep(new PrimType(*this, PrimType_##T, 1)))
#include "anydsl2/tables/primtypetable.h"
{}

World::~World() {
    for_all (primop, primops_) delete primop;
    for_all (type,   types_  ) delete type;
    for_all (lambda, lambdas_) delete lambda;
}

const Type* World::keep_nocast(const Type* type) {
    std::pair<TypeSet::iterator, bool> tp = types_.insert(type);
    assert(tp.second);
    typekeeper(type);
    return type;
}

/*
 * types
 */

Sigma* World::named_sigma(size_t size, const std::string& name) {
    Sigma* s = new Sigma(*this, size, name);
    assert(types_.find(s) == types_.end() && "must not be inside");
    types_.insert(s).first;
    return s;
}

const Opaque* World::opaque(ArrayRef<const Type*> types, ArrayRef<uint32_t> flags) { return unify(new Opaque(*this, types, flags)); }

/*
 * literals
 */

const Def* World::literal(PrimTypeKind kind, int value, size_t length) {
    const Def* lit;
    if (length == 1) {
        switch (kind) {
#define ANYDSL2_U_TYPE(T) case PrimType_##T: lit = literal(T(value), 1); break;
#define ANYDSL2_F_TYPE(T) ANYDSL2_U_TYPE(T)
#include "anydsl2/tables/primtypetable.h"
            default: ANYDSL2_UNREACHABLE;
        }
    }

    return vector(lit, length);
}

const Def* World::literal(PrimTypeKind kind, Box box, size_t length) { return vector(cse(new PrimLit(*this, kind, box, "")), length); }
const Def* World::any    (const Type* type, size_t length) { return vector(cse(new Any(type, "")), length); }
const Def* World::bottom (const Type* type, size_t length) { return vector(cse(new Bottom(type, "")), length); }
const Def* World::zero   (const Type* type, size_t length) { return zero  (type->as<PrimType>()->primtype_kind(), length); }
const Def* World::one    (const Type* type, size_t length) { return one   (type->as<PrimType>()->primtype_kind(), length); }
const Def* World::allset (const Type* type, size_t length) { return allset(type->as<PrimType>()->primtype_kind(), length); }
const TypeKeeper* World::typekeeper(const Type* type, const std::string& name) { return cse(new TypeKeeper(type, name)); }

/*
 * create
 */

const Def* World::binop(int kind, const Def* lhs, const Def* rhs, const std::string& name) {
    if (is_arithop(kind))
        return arithop((ArithOpKind) kind, lhs, rhs);

    assert(is_relop(kind) && "must be a RelOp");
    return relop((RelOpKind) kind, lhs, rhs);
}

const Def* World::arithop(ArithOpKind kind, const Def* a, const Def* b, const std::string& name) {
    assert(a->type() == b->type());
    assert(a->type()->as<PrimType>()->length() == b->type()->as<PrimType>()->length());
    PrimTypeKind type = a->type()->as<PrimType>()->primtype_kind();

    // bottom op bottom -> bottom
    if (a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(type);

    const PrimLit* llit = a->isa<PrimLit>();
    const PrimLit* rlit = b->isa<PrimLit>();
    const Vector*  lvec = a->isa<Vector>();
    const Vector*  rvec = b->isa<Vector>();

    if (lvec && rvec) {
        size_t num = lvec->type()->as<PrimType>()->length();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = arithop(kind, lvec->op(i), rvec->op(i));
        return vector(ops, name);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();

        switch (kind) {
            case ArithOp_add:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_sub:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_mul:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_udiv:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: \
                        return rlit->is_zero() \
                             ? (const Def*) bottom(type) \
                             : literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_sdiv:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return rlit->is_zero() \
                            ? (const Def*) bottom(type) \
                            : literal(type, Box(bcast<T , S>(bcast<S, T >(l.get_##T()) / bcast<S, T >(r.get_##T())))); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_urem:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: \
                        return rlit->is_zero() \
                             ? (const Def*) bottom(type) \
                             : literal(type, Box(T(l.get_##T() % r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_srem:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return literal(type, Box(bcast<T , S>(bcast<S, T >(l.get_##T()) % bcast<S, T >(r.get_##T())))); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_and:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_or:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_xor:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_shl:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() << r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_lshr:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() >> r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_ashr:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return literal(type, Box(bcast<T , S>(bcast<S, T >(l.get_##T()) >> bcast<S, T >(r.get_##T())))); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_fadd:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_fsub:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_fmul:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_fdiv:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_frem:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(std::fmod(l.get_##T(), r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
        }
    }

    if (a == b) {
        switch (kind) {
            case ArithOp_add:  return arithop_mul(literal(type, 2), a);

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

    if (rlit) {
        if (rlit->is_zero()) {
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
        } else if (rlit->is_one()) {
            switch (kind) {
                case ArithOp_sdiv:
                case ArithOp_udiv: return a;

                case ArithOp_srem:
                case ArithOp_urem: return zero(type);

                default: break;
            }
        } else if (rlit->primlit_value<uint64_t>() >= uint64_t(num_bits(type))) {
            switch (kind) {
                case ArithOp_shl:
                case ArithOp_ashr:
                case ArithOp_lshr: return bottom(type);

                default: break;
            }
        }
    }

    // do we have ~~x?
    if (kind == ArithOp_xor && a->is_allset() && b->is_not())
        return b->as<ArithOp>()->rhs();

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

    if (llit) {
        if (llit->is_zero()) {
            switch (kind) {
                case ArithOp_mul:
                case ArithOp_sdiv:
                case ArithOp_udiv:
                case ArithOp_srem:
                case ArithOp_urem:
                case ArithOp_and:
                case ArithOp_shl:
                case ArithOp_lshr:
                case ArithOp_ashr: return zero(type);

                case ArithOp_add: 
                case ArithOp_or:
                case ArithOp_xor:  return b;

                default: break;
            }
        } else if (llit->is_one()) {
            switch (kind) {
                case ArithOp_mul: return b;
                default: break;
            }
        } else if (llit->is_allset()) {
            switch (kind) {
                case ArithOp_and: return b;
                case ArithOp_or:  return llit; // allset
                default: break;
            }
        }
    }

    // normalize: try to reorder same ops to have the literal/vector on the left-most side
    if (is_associative(kind)) {
        const ArithOp* a_same = a->isa<ArithOp>() && a->as<ArithOp>()->arithop_kind() == kind ? a->as<ArithOp>() : 0;
        const ArithOp* b_same = b->isa<ArithOp>() && b->as<ArithOp>()->arithop_kind() == kind ? b->as<ArithOp>() : 0;
        const Def* a_lhs_lv = a_same && (a_same->lhs()->isa<PrimLit>() || a_same->lhs()->isa<Vector>()) ? a_same->lhs() : 0;
        const Def* b_lhs_lv = b_same && (b_same->lhs()->isa<PrimLit>() || b_same->lhs()->isa<Vector>()) ? b_same->lhs() : 0;

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

    return cse(new ArithOp(kind, a, b, name));
}

const Def* World::arithop_not(const Def* def) { return arithop_xor(allset(def->type()), def); }

const Def* World::arithop_minus(const Def* def) {
    const Def* zero;
    switch (PrimTypeKind kind = def->type()->as<PrimType>()->primtype_kind()) {
        case PrimType_f32: zero = literal_f32(-0.f); break;
        case PrimType_f64: zero = literal_f64(-0.0); break;
        default: assert(is_int(kind)); zero = this->zero(kind);
    }
    return arithop_sub(zero, def);
}

const Def* World::relop(RelOpKind kind, const Def* a, const Def* b, const std::string& name) {
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

    const PrimLit* llit = a->isa<PrimLit>();
    const PrimLit* rlit = b->isa<PrimLit>();
    const Vector*  lvec = b->isa<Vector>();
    const Vector*  rvec = b->isa<Vector>();

    if (lvec && rvec) {
        size_t num = lvec->type()->as<PrimType>()->length();
        Array<const Def*> ops(num);
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
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_ne:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_ult:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_ule:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_slt:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed< T >::type S; \
                        return literal_u1(bcast<S, T>(l.get_##T()) <  bcast<S, T>(r.get_##T())); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_sle:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed< T >::type S; \
                        return literal_u1(bcast<S, T>(l.get_##T()) <= bcast<S, T>(r.get_##T())); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_fcmp_oeq:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case RelOp_fcmp_one:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case RelOp_fcmp_olt:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case RelOp_fcmp_ole:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            default:
                ANYDSL2_UNREACHABLE;
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

    return cse(new RelOp(kind, a, b, name));
}

const Def* World::convop(ConvOpKind kind, const Def* from, const Type* to, const std::string& name) {
    if (from->isa<Bottom>())
        return bottom(to);

#if 0
    if (const PrimLit* lit = from->isa<PrimLit>())
        Box box = lit->box();
        PrimTypeKind type = lit->primtype_kind();

        // TODO folding
    }
#endif

    return cse(new ConvOp(kind, from, to, name));
}

const Def* World::tuple_extract(const Def* agg, const Def* index, const std::string& name) {
    if (agg->isa<Bottom>())
        return bottom(agg->type()->as<Sigma>()->elem_via_lit(index));

    if (const Tuple* tuple = agg->isa<Tuple>())
        return tuple->op_via_lit(index);

    if (const TupleInsert* insert = agg->isa<TupleInsert>()) {
        if (index == insert->index())
            return insert->value();
        else
            return tuple_extract(insert->tuple(), index);
    }

    return cse(new TupleExtract(agg, index, name));
}

const Def* World::tuple_insert(const Def* agg, const Def* index, const Def* value, const std::string& name) {
    if (agg->isa<Bottom>() || value->isa<Bottom>())
        return bottom(agg->type());

    if (const Tuple* tup = agg->isa<Tuple>()) {
        Array<const Def*> args(tup->size());
        std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
        args[index->primlit_value<size_t>()] = value;

        return tuple(args);
    }

    return cse(new TupleInsert(agg, index, value, name));
}

const Def* World::tuple_extract(const Def* tuple, u32 index, const std::string& name) { 
    return tuple_extract(tuple, literal_u32(index), name); 
}
const Def* World::tuple_insert(const Def* tuple, u32 index, const Def* value, const std::string& name) { 
    return tuple_insert(tuple, literal_u32(index), value, name); 
}

const Def* World::vector(const Def* arg, size_t length, const std::string& name) {
    if (length == 1) 
        return arg;

    Array<const Def*> args(length);
    std::fill(args.begin(), args.end(), arg);
    return vector(args, name);
}

const Enter* World::enter(const Def* mem, const std::string& name) {
    if (const Leave* leave = mem->isa<Leave>())
        if (const TupleExtract* extract = leave->frame()->isa<TupleExtract>())
            if (const Enter* old_enter = extract->tuple()->isa<Enter>())
                return old_enter;

    if (const TupleExtract* extract = mem->isa<TupleExtract>())
        if (const Enter* old_enter = extract->tuple()->isa<Enter>())
            return old_enter;

    return cse(new Enter(mem, name));
}

const Def* World::select(const Def* cond, const Def* a, const Def* b, const std::string& name) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type());

    if (const PrimLit* lit = cond->isa<PrimLit>())
        return lit->value().get_u1().get() ? a : b;

    if (cond->is_not()) {
        cond = cond->as<ArithOp>()->rhs();
        std::swap(a, b);
    }

    return cse(new Select(cond, a, b, name));
}

const Load* World::load(const Def* mem, const Def* ptr, const std::string& name) { 
    return cse(new Load(mem, ptr, name)); 
}
const Store* World::store(const Def* mem, const Def* ptr, const Def* value, const std::string& name) {
    return cse(new Store(mem, ptr, value, name));
}
const Leave* World::leave(const Def* mem, const Def* frame, const std::string& name) { 
    return cse(new Leave(mem, frame, name)); }
const Slot* World::slot(const Type* type, const Def* frame, size_t index, const std::string& name) {
    return cse(new Slot(type, frame, index, name));
}
const LEA* World::lea(const Def* ptr, const Def* index, const std::string& name) { return cse(new LEA(ptr, index, name)); }

Lambda* World::lambda(const Pi* pi, LambdaAttr attr, const std::string& name) {
    ANYDSL2_CHECK_BREAK
    Lambda* l = new Lambda(gid_++, pi, attr, true, name);
    lambdas_.insert(l);

    size_t i = 0;
    for_all (elem, pi->elems())
        l->params_.push_back(param(elem, l, i++));

    return l;
}

Lambda* World::basicblock(const std::string& name) {
    ANYDSL2_CHECK_BREAK
    Lambda* bb = new Lambda(gid_++, pi0(), LambdaAttr(0), false, name);
    lambdas_.insert(bb);
    return bb;
}

const Def* World::rebuild(const PrimOp* in, ArrayRef<const Def*> ops) {
    int kind = in->kind();
    const Type* type = in->type();
    const std::string& name = in->name;

    if (ops.empty()) return in;
    if (is_arithop(kind)) { assert(ops.size() == 2); return arithop((ArithOpKind) kind, ops[0], ops[1], name); }
    if (is_relop  (kind)) { assert(ops.size() == 2); return relop(  (RelOpKind  ) kind, ops[0], ops[1], name); }
    if (is_convop (kind)) { assert(ops.size() == 1); return convop( (ConvOpKind ) kind, ops[0],   type, name); }

    switch (kind) {
        case Node_Enter:   assert(ops.size() == 1); return enter(  ops[0], name);
        case Node_Leave:   assert(ops.size() == 2); return leave(  ops[0], ops[1], name);
        case Node_Load:    assert(ops.size() == 2); return load(   ops[0], ops[1], name);
        case Node_Select:  assert(ops.size() == 3); return select( ops[0], ops[1], ops[2], name);
        case Node_Store:   assert(ops.size() == 3); return store(  ops[0], ops[1], ops[2], name);
        case Node_Tuple:                            return tuple(  ops, name);
        case Node_TupleExtract: assert(ops.size() == 2); return tuple_extract(ops[0], ops[1], name);
        case Node_TupleInsert:  assert(ops.size() == 3); return tuple_insert( ops[0], ops[1], ops[2], name);
        case Node_Slot:    assert(ops.size() == 1); 
            return slot(type->as<Ptr>()->referenced_type(), ops[0], in->as<Slot>()->index(), name);
        default: ANYDSL2_UNREACHABLE;
    }
}

const Type* World::rebuild(const Type* type, ArrayRef<const Type*> elems) {
    if (elems.empty()) return type;

    switch (type->kind()) {
        case Node_Pi:    return pi(elems);
        case Node_Sigma: return sigma(elems);
        case Node_Ptr:   assert(elems.size() == 1); return ptr(elems.front());
        default: ANYDSL2_UNREACHABLE;
    }
}

const Param* World::param(const Type* type, Lambda* lambda, size_t index, const std::string& name) {
    ANYDSL2_CHECK_BREAK
    return new Param(gid_++, type, lambda, index, name);
}

/*
 * cse + unify
 */

const Type* World::unify_base(const Type* type) {
    TypeSet::iterator i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        return *i;
    }

    std::pair<TypeSet::iterator, bool> p = types_.insert(type);
    assert(p.second && "hash/equal broken");
    return type;
}

const Def* World::cse_base(const PrimOp* primop) {
    PrimOpSet::iterator i = primops_.find(primop);
    if (i != primops_.end()) {
        for (size_t x = 0, e = primop->size(); x != e; ++x)
            primop->unregister_use(x);

        delete primop;
        primop = *i;
    } else {
        std::pair<PrimOpSet::iterator, bool> p = primops_.insert(primop);
        assert(p.second && "hash/equal broken");
        primop->set_gid(gid_++);
    }

    if (breakpoints_.find(primop->gid()) != breakpoints_.end()) 
        ANYDSL2_BREAK

    return primop;
}

/*
 * optimizations
 */

template<class S>
void World::wipe_out(const size_t pass, S& set) {
    for (typename S::iterator i = set.begin(); i != set.end();) {
        typename S::iterator j = i++;
        const Def* def = *j;
        if (!def->is_visited(pass)) {
            delete def;
            set.erase(j);
        }
    }
}

template<class S>
void World::unregister_uses(const size_t pass, S& set) {
    for (typename S::iterator i = set.begin(), e = set.end(); i != e; ++i) {
        const Def* def = *i;
        if (!def->is_visited(pass)) {
            for (size_t i = 0, e = def->size(); i != e; ++i) {
                def->unregister_use(i);
            }
        }
    }
}

void World::unreachable_code_elimination() {
    const size_t pass = new_pass();

    for_all (lambda, lambdas())
        if (lambda->attr().is_extern())
            uce_insert(pass, lambda);

    for_all (lambda, lambdas()) {
        if (!lambda->is_visited(pass))
            lambda->destroy_body();
    }
}

void World::uce_insert(const size_t pass, Lambda* lambda) {
    if (lambda->visit(pass)) return;

    for_all (succ, lambda->succs())
        uce_insert(pass, succ);
}

void World::dead_code_elimination() {
    const size_t pass = new_pass();

    for_all (primop, primops()) {
        if (const TypeKeeper* tk = primop->isa<TypeKeeper>())
            dce_insert(pass, tk);
    }

    for_all (lambda, lambdas()) {
        if (lambda->attr().is_extern()) {
            if (lambda->empty()) {
                for_all (param, lambda->params())
                    dce_insert(pass, param);
            } else {
                for_all (param, lambda->params()) {
                    if (param->order() >= 1) {
                        for_all (use, param->uses()) {
                            if (Lambda* caller = use->isa_lambda())
                                dce_insert(pass, caller);
                        }
                    }
                }
            }
        }
    }

    for_all (lambda, lambdas()) {
        if (!lambda->is_visited(pass))
            lambda->destroy_body();
        else {
            for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                const Def* arg = lambda->arg(i);
                if (!arg->is_visited(pass)) {
                    const Def* bot = bottom(arg->type());
                    bot->visit(pass);
                    lambda->update_arg(i, bot);
                }
            }
        }
    }

    unregister_uses(pass, primops_);
    unregister_uses(pass, lambdas_);
    wipe_out(pass, primops_);
    wipe_out(pass, lambdas_);
}

void World::dce_insert(const size_t pass, const Def* def) {
#ifndef NDEBUG
    if (const PrimOp* primop = def->isa<PrimOp>()) assert(primops_.find(primop)          != primops_.end());
    if (      Lambda* lambda = def->isa_lambda() ) assert(lambdas_.find(lambda)          != lambdas_.end());
    if (const Param*  param  = def->isa<Param>() ) assert(lambdas_.find(param->lambda()) != lambdas_.end());
#endif

    if (def->visit(pass)) return;

    if (const PrimOp* primop = def->isa<PrimOp>()) {
        for_all (op, primop->ops())
            dce_insert(pass, op);
    } else if (const Param* param = def->isa<Param>()) {
        for_all (peek, param->peek())
            dce_insert(pass, peek.def());
    } else {
        Lambda* lambda = def->as_lambda();
        for_all (pred, lambda->preds()) // insert control-dependent lambdas
            dce_insert(pass, pred);
        if (!lambda->empty()) {
            dce_insert(pass, lambda->to());
            if (lambda->to()->isa<Param>()) {
                for_all (arg, lambda->args())
                    dce_insert(pass, arg);
            }
        }
    }
}

void World::unused_type_elimination() {
    const size_t pass = new_pass();

    for_all (primop, primops())
        ute_insert(pass, primop->type());

    for_all (lambda, lambdas()) {
        ute_insert(pass, lambda->type());
        for_all (param, lambda->params())
            ute_insert(pass, param->type());
    }

    for (TypeSet::iterator i = types_.begin(); i != types_.end();) {
        const Type* type = *i;
        if (type->is_visited(pass))
            ++i;
        else {
            delete type;
            i = types_.erase(i);
        }
    }
}

void World::ute_insert(const size_t pass, const Type* type) {
    assert(types_.find(type) != types_.end() && "not in map");

    if (type->visit(pass)) return;

    for_all (elem, type->elems())
        ute_insert(pass, elem);
}

void World::cleanup() {
    unreachable_code_elimination();
    dead_code_elimination();
    unused_type_elimination();
}

void World::opt() {
    cleanup();
    assert(verify(*this)); mem2reg(*this); cleanup();
    assert(verify(*this)); cfg_transform(*this); cleanup();
    assert(verify(*this)); inliner(*this);       cleanup();
    assert(verify(*this)); merge_lambdas(*this); cleanup();
}

PrimOp* World::release(const PrimOp* primop) {
    PrimOpSet::iterator i = primops_.find(primop);
    assert(i != primops_.end() && "must be found");
    assert(primop == *i);
    primops_.erase(i);

    return const_cast<PrimOp*>(primop);
}

void World::reinsert(const PrimOp* primop) {
    assert(primops_.find(primop) == primops_.end() && "must not be found");
    primops_.insert(primop);
}

} // namespace anydsl2
