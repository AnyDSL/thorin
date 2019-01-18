#include "thorin/world.h"

#include <fstream>

#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/clone_bodies.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/transform/dead_load_opt.h"
#include "thorin/transform/flatten_tuples.h"
#include "thorin/transform/rewrite_flow_graphs.h"
#include "thorin/transform/hoist_enters.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/lift_builtins.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/transform/split_slots.h"
#include "thorin/util/array.h"
#include "thorin/util/log.h"

namespace thorin {

/*
 * constructor and destructor
 */

World::World(std::string name)
    : name_(name)
    , universe_ (insert(new Universe(*this)))
    , star_     (insert(new Kind(*this, Node_Star)))
    , sigma_    (insert(new Sigma(star_, Defs{}, {"[]"})))
    , bottom_   (insert(new Bottom(star_, {"<⊥:*>"})))
    , mem_      (insert(new MemType  (*this)))
    , frame_    (insert(new FrameType(*this)))
#define THORIN_ALL_TYPE(T, M) \
    , T##_      (insert(new PrimType(*this, PrimType_##T, 1, {#T})))
#include "thorin/tables/primtypetable.h"
    , branch_   (lam(cn(sigma({type_bool(), cn(), cn()})), CC::C, Intrinsic::Branch, {"br"}))
    , end_scope_(lam(cn(), CC::C, Intrinsic::EndScope, {"end_scope"}))
{}

World::~World() {
    for (auto def : defs_) delete def;
}

const Def* World::app(const Def* callee, const Def* arg, Debug dbg) {
    auto pi = callee->type()->as<Pi>();
    assertf(pi->domain() == arg->type(), "'{}' is if of type '{}' but calls '{}' of type '{}'\n", callee, pi, arg, arg->type());

    if (auto lam = callee->isa<Lam>()) {
        switch (lam->intrinsic()) {
            case Intrinsic::Branch: {
                auto cond = extract(arg, 0_s);
                auto t    = extract(arg, 1_s);
                auto f    = extract(arg, 2_s);
                if (auto lit = cond->isa<PrimLit>())
                    return app(lit->value().get_bool() ? t : f, Defs{}, dbg);
                if (t == f)
                    return app(t, Defs{}, dbg);
                if (is_not(cond))
                    return branch(cond->as<ArithOp>()->rhs(), f, t, dbg);
                break;
            }
            case Intrinsic::Match: {
                auto args = arg->as<Tuple>()->ops();
                if (args.size() == 2) return app(args[1], Defs{}, dbg);
                if (auto lit = args[0]->isa<PrimLit>()) {
                    for (size_t i = 2; i < args.size(); i++) {
                        if (extract(args[i], 0_s)->as<PrimLit>() == lit)
                            return app(extract(args[i], 1), Defs{}, dbg);
                    }
                    return app(args[1], Defs{}, dbg);
                }
                break;
            }
            default:
                break;
        }
    }

    return unify(new App(pi->codomain(), callee, arg, dbg));
}

const Lam* World::lam(const Def* domain, const Def* filter, const Def* body, Debug dbg) {
    auto p = pi(domain, body->type());
    return unify(new Lam(p, filter, body, dbg));
}

const Pi* World::pi(const Def* domain, const Def* codomain, Debug dbg) {
    auto type = star(); // TODO
    return unify(new Pi(type, domain, codomain, dbg));
}

const Def* World::sigma(const Def* type, Defs ops, Debug dbg) {
    return ops.size() == 1 ? ops.front() : unify(new Sigma(type, ops, dbg));
}

static const Def* infer_sigma(World& world, Defs ops) {
    Array<const Def*> elems(ops.size());
    for (size_t i = 0, e = ops.size(); i != e; ++i)
        elems[i] = ops[i]->type();

    return world.sigma(elems);
}

const Def* World::tuple(Defs ops, Debug dbg) {
    return tuple(infer_sigma(*this, ops), ops, dbg);
}

const Def* World::tuple(const Def* type, Defs ops, Debug dbg) {
#if THORIN_ENABLE_CHECKS
    // TODO type-check type vs inferred type
#endif
    if (type->is_nominal()) return unify(new Tuple(type, ops, dbg));
    return ops.size() == 1 ? ops.front() : try_fold_aggregate(unify(new Tuple(type, ops, dbg)));
}

/*
 * literals
 */

const Def* World::splat(const Def* arg, size_t length, Debug dbg) {
    if (length == 1)
        return arg;

    Array<const Def*> args(length);
    std::fill(args.begin(), args.end(), arg);
    return vector(args, dbg);
}

const Def* World::allset(PrimTypeTag tag, Debug dbg, size_t length) {
    switch (tag) {
#define THORIN_I_TYPE(T, M) \
    case PrimType_##T: return literal(PrimType_##T, Box(~T(0)), dbg, length);
#define THORIN_BOOL_TYPE(T, M) \
    case PrimType_##T: return literal(PrimType_##T, Box(true), dbg, length);
#include "thorin/tables/primtypetable.h"
        default: THORIN_UNREACHABLE;
    }
}

/*
 * arithops
 */

const Def* World::binop(int tag, const Def* lhs, const Def* rhs, Debug dbg) {
    if (is_arithop(tag))
        return arithop((ArithOpTag) tag, lhs, rhs, dbg);

    assert(is_cmp(tag) && "must be a Cmp");
    return cmp((CmpTag) tag, lhs, rhs, dbg);
}

const Def* World::arithop(ArithOpTag tag, const Def* a, const Def* b, Debug dbg) {
    assert(a->type() == b->type());
    assert(a->type()->as<PrimType>()->length() == b->type()->as<PrimType>()->length());
    PrimTypeTag type = a->type()->as<PrimType>()->primtype_tag();

    auto llit = a->isa<PrimLit>();
    auto rlit = b->isa<PrimLit>();
    auto lvec = a->isa<Vector>();
    auto rvec = b->isa<Vector>();

    if (lvec && rvec) {
        size_t num = lvec->type()->as<PrimType>()->length();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = arithop(tag, lvec->op(i), rvec->op(i), dbg);
        return vector(ops, dbg);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();

        try {
            switch (tag) {
                case ArithOp_add:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_sub:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_mul:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_div:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_rem:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() % r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_and:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())), dbg);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_or:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())), dbg);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_xor:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())), dbg);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shl:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() << r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shr:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() >> r.get_##T())), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
            }
        } catch (BottomException) {
            return bottom(type, dbg);
        }
    }

    // normalize: swap literal/vector to the left
    if (is_commutative(tag) && (rlit || rvec)) {
        std::swap(a, b);
        std::swap(llit, rlit);
        std::swap(lvec, rvec);
    }

    if (is_type_i(type) || type == PrimType_bool) {
        if (a == b) {
            switch (tag) {
                case ArithOp_add: return arithop_mul(literal(type, 2, dbg), a, dbg);

                case ArithOp_sub:
                case ArithOp_xor: return zero(type, dbg);

                case ArithOp_and:
                case ArithOp_or:  return a;

                case ArithOp_div:
                    if (is_zero(b))
                        return bottom(type, dbg);
                    return one(type, dbg);

                case ArithOp_rem:
                    if (is_zero(b))
                        return bottom(type, dbg);
                    return zero(type, dbg);

                default: break;
            }
        }

        if (is_zero(a)) {
            switch (tag) {
                case ArithOp_mul:
                case ArithOp_div:
                case ArithOp_rem:
                case ArithOp_and:
                case ArithOp_shl:
                case ArithOp_shr: return zero(type, dbg);

                case ArithOp_add:
                case ArithOp_or:
                case ArithOp_xor:  return b;

                default: break;
            }
        }

        if (is_one(a)) {
            switch (tag) {
                case ArithOp_mul: return b;
                default: break;
            }
        }

        if (is_allset(a)) {
            switch (tag) {
                case ArithOp_and: return b;
                case ArithOp_or:  return llit; // allset
                default: break;
            }
        }

        if (is_zero(b)) {
            switch (tag) {
                case ArithOp_div:
                case ArithOp_rem: return bottom(type, dbg);

                case ArithOp_shl:
                case ArithOp_shr: return a;

                default: break;
            }
        }

        if (is_one(b)) {
            switch (tag) {
                case ArithOp_mul:
                case ArithOp_div: return a;
                case ArithOp_rem: return zero(type, dbg);

                default: break;
            }
        }

        if (rlit && primlit_value<uint64_t>(rlit) >= uint64_t(num_bits(type))) {
            switch (tag) {
                case ArithOp_shl:
                case ArithOp_shr: return bottom(type, dbg);

                default: break;
            }
        }

        if (tag == ArithOp_xor && is_allset(a)) {    // is this a NOT
            if (is_not(b))                            // do we have ~~x?
                return b->as<ArithOp>()->rhs();
            if (auto cmp = b->isa<Cmp>())   // do we have ~(a cmp b)?
                return this->cmp(negate(cmp->cmp_tag()), cmp->lhs(), cmp->rhs(), dbg);
        }

        auto lcmp = a->isa<Cmp>();
        auto rcmp = b->isa<Cmp>();

        if (tag == ArithOp_or && lcmp && rcmp && lcmp->lhs() == rcmp->lhs() && lcmp->rhs() == rcmp->rhs()
                && lcmp->cmp_tag() == negate(rcmp->cmp_tag()))
                return literal_bool(true, dbg);

        if (tag == ArithOp_and && lcmp && rcmp && lcmp->lhs() == rcmp->lhs() && lcmp->rhs() == rcmp->rhs()
                && lcmp->cmp_tag() == negate(rcmp->cmp_tag()))
                return literal_bool(false, dbg);

        auto land = a->tag() == Node_and ? a->as<ArithOp>() : nullptr;
        auto rand = b->tag() == Node_and ? b->as<ArithOp>() : nullptr;

        // distributivity (a and b) or (a and c)
        if (tag == ArithOp_or && land && rand) {
            if (land->lhs() == rand->lhs())
                return arithop_and(land->lhs(), arithop_or(land->rhs(), rand->rhs(), dbg), dbg);
            if (land->rhs() == rand->rhs())
                return arithop_and(land->rhs(), arithop_or(land->lhs(), rand->lhs(), dbg), dbg);
        }

        auto lor = a->tag() == Node_or ? a->as<ArithOp>() : nullptr;
        auto ror = b->tag() == Node_or ? b->as<ArithOp>() : nullptr;

        // distributivity (a or b) and (a or c)
        if (tag == ArithOp_and && lor && ror) {
            if (lor->lhs() == ror->lhs())
                return arithop_or(lor->lhs(), arithop_and(lor->rhs(), ror->rhs(), dbg), dbg);
            if (lor->rhs() == ror->rhs())
                return arithop_or(lor->rhs(), arithop_and(lor->lhs(), ror->lhs(), dbg), dbg);
        }

        // absorption: a and (a or b) = a
        if (tag == ArithOp_and) {
            if (ror) {
                if (a == ror->lhs()) return ror->rhs();
                if (a == ror->rhs()) return ror->lhs();
            }
            if (lor) {
                if (a == lor->lhs()) return lor->rhs();
                if (a == lor->rhs()) return lor->lhs();
            }
        }

        // absorption: a or (a and b) = a
        if (tag == ArithOp_or) {
            if (rand) {
                if (a == rand->lhs()) return rand->rhs();
                if (a == rand->rhs()) return rand->lhs();
            }
            if (land) {
                if (a == land->lhs()) return land->rhs();
                if (a == land->rhs()) return land->lhs();
            }
        }

        if (tag == ArithOp_or) {
            if (lor && ror) {
                if (lor->lhs() == ror->lhs())
                    return arithop_or(lor->rhs(), ror->rhs(), dbg);
                if (lor->rhs() == ror->rhs())
                    return arithop_or(lor->lhs(), ror->lhs(), dbg);
            }
        }

        if (tag == ArithOp_and) {
            if (land && rand) {
                if (land->lhs() == rand->lhs())
                    return arithop_and(land->rhs(), rand->rhs(), dbg);
                if (land->rhs() == rand->rhs())
                    return arithop_and(land->lhs(), rand->lhs(), dbg);
            }
        }
    }

    // normalize: try to reorder same ops to have the literal/vector on the left-most side
    if (is_associative(tag) && is_type_i(a->type())) {
        auto a_same = a->isa<ArithOp>() && a->as<ArithOp>()->arithop_tag() == tag ? a->as<ArithOp>() : nullptr;
        auto b_same = b->isa<ArithOp>() && b->as<ArithOp>()->arithop_tag() == tag ? b->as<ArithOp>() : nullptr;
        auto a_lhs_lv = a_same && (a_same->lhs()->isa<PrimLit>() || a_same->lhs()->isa<Vector>()) ? a_same->lhs() : nullptr;
        auto b_lhs_lv = b_same && (b_same->lhs()->isa<PrimLit>() || b_same->lhs()->isa<Vector>()) ? b_same->lhs() : nullptr;

        if (is_commutative(tag)) {
            if (a_lhs_lv && b_lhs_lv)
                return arithop(tag, arithop(tag, a_lhs_lv, b_lhs_lv, dbg), arithop(tag, a_same->rhs(), b_same->rhs(), dbg), dbg);
            if ((llit || lvec) && b_lhs_lv)
                return arithop(tag, arithop(tag, a, b_lhs_lv, dbg), b_same->rhs(), dbg);
            if (b_lhs_lv)
                return arithop(tag, b_lhs_lv, arithop(tag, a, b_same->rhs(), dbg), dbg);
        }
        if (a_lhs_lv)
            return arithop(tag, a_lhs_lv, arithop(tag, a_same->rhs(), b, dbg), dbg);
    }

    return unify(new ArithOp(tag, a, b, dbg));
}

const Def* World::arithop_not(const Def* def, Debug dbg) { return arithop_xor(allset(def->type(), dbg, vector_length(def)), def, dbg); }

const Def* World::arithop_minus(const Def* def, Debug dbg) {
    switch (PrimTypeTag tag = def->type()->as<PrimType>()->primtype_tag()) {
#define THORIN_F_TYPE(T, M) \
        case PrimType_##T: \
            return arithop_sub(literal_##T(M(-0.f), dbg, vector_length(def)), def, dbg);
#include "thorin/tables/primtypetable.h"
        default:
            assert(is_type_i(tag));
            return arithop_sub(zero(tag, dbg), def, dbg);
    }
}

/*
 * compares
 */

const Def* World::cmp(CmpTag tag, const Def* a, const Def* b, Debug dbg) {
    CmpTag oldtag = tag;
    switch (tag) {
        case Cmp_gt: tag = Cmp_lt; break;
        case Cmp_ge: tag = Cmp_le; break;
        default: break;
    }

    if (a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(type_bool(), dbg);

    if (oldtag != tag)
        std::swap(a, b);

    auto llit = a->isa<PrimLit>();
    auto rlit = b->isa<PrimLit>();
    auto lvec = a->isa<Vector>();
    auto rvec = b->isa<Vector>();

    if (lvec && rvec) {
        size_t num = lvec->type()->as<PrimType>()->length();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cmp(tag, lvec->op(i), rvec->op(i), dbg);
        return vector(ops, dbg);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();
        PrimTypeTag type = llit->primtype_tag();

        switch (tag) {
            case Cmp_eq:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() == r.get_##T(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case Cmp_ne:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() != r.get_##T(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case Cmp_lt:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() <  r.get_##T(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case Cmp_le:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() <= r.get_##T(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            default: THORIN_UNREACHABLE;
        }
    }

    if (a == b) {
        switch (tag) {
            case Cmp_lt:
            case Cmp_ne: return zero(type_bool(), dbg);
            case Cmp_le:
            case Cmp_eq: return one(type_bool(), dbg);
            default: break;
        }
    }

    return unify(new Cmp(tag, a, b, dbg));
}

/*
 * casts
 */

const Def* World::convert(const Def* dst_type, const Def* src, Debug dbg) {
    if (dst_type == src->type())
        return src;
    if (src->type()->isa<PtrType>() && dst_type->isa<PtrType>())
        return bitcast(dst_type, src, dbg);
    if (auto dst_sigma = dst_type->isa<Sigma>()) {
        assert(dst_sigma->num_ops() == src->type()->as<Sigma>()->num_ops());

        Array<const Def*> new_tuple(dst_sigma->num_ops());
        for (size_t i = 0, e = new_tuple.size(); i != e; ++i)
            new_tuple[i] = convert(dst_type->op(i), extract(src, i, dbg), dbg);

        return tuple(dst_sigma, new_tuple, dbg);
    }

    return cast(dst_type, src, dbg);
}

const Def* World::cast(const Def* to, const Def* from, Debug dbg) {
    if (from->isa<Bottom>())
        return bottom(to);

    if (from->type() == to)
        return from;

    if (auto vec = from->isa<Vector>()) {
        size_t num = vector_length(vec);
        auto to_vec = to->as<VectorType>();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cast(to_vec->scalarize(), vec->op(i), dbg);
        return vector(ops, dbg);
    }

    if (auto variant = from->isa<Variant>()) {
        if (variant->op(0)->type() != to)
            ELOG("variant downcast not possible");
        return variant->op(0);
    }

    auto lit = from->isa<PrimLit>();
    auto to_type = to->isa<PrimType>();
    if (lit && to_type) {
        Box box = lit->value();

        switch (lit->primtype_tag()) {
            case PrimType_bool:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_bool()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps8:
            case PrimType_qs8:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s8()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps16:
            case PrimType_qs16:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s16()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps32:
            case PrimType_qs32:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s32()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps64:
            case PrimType_qs64:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s64()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu8:
            case PrimType_qu8:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u8()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu16:
            case PrimType_qu16:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u16()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu32:
            case PrimType_qu32:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u32()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu64:
            case PrimType_qu64:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u64()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pf16:
            case PrimType_qf16:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_f16()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pf32:
            case PrimType_qf32:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_f32()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pf64:
            case PrimType_qf64:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_f64()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
        }
    }

    return unify(new Cast(to, from, dbg));
}

const Def* World::bitcast(const Def* to, const Def* from, Debug dbg) {
    if (from->isa<Bottom>())
        return bottom(to);

    if (from->type() == to)
        return from;

    if (auto other = from->isa<Bitcast>()) {
        // reduce bitcast chains
        do {
            auto value = other->from();
            if (to == value->type())
                return value;
            other = value->isa<Bitcast>();
        } while (other);
    }

    auto prim_to = to->isa<PrimType>();
    auto prim_from = from->type()->isa<PrimType>();
    if (prim_to && prim_from) {
        if (num_bits(prim_from->primtype_tag()) != num_bits(prim_to->primtype_tag()))
            ELOG("bitcast between primitive types of different size");
        // constant folding
        if (auto lit = from->isa<PrimLit>())
            return literal(prim_to->primtype_tag(), lit->value(), dbg);
    }

    if (auto vec = from->isa<Vector>()) {
        size_t num = vector_length(vec);
        auto to_vec = to->as<VectorType>();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = bitcast(to_vec->scalarize(), vec->op(i), dbg);
        return vector(ops, dbg);
    }

    return unify(new Bitcast(to, from, dbg));
}

/*
 * aggregate operations
 */

static bool fold_1_tuple(const Def* type, const Def* index) {
    if (auto lit = index->isa<PrimLit>()) {
        if (primlit_value<u64>(lit) == 0
                && !type->isa<ArrayType>()
                && !type->isa<Sigma>()) {
            if (auto prim_type = type->isa<PrimType>())
                return prim_type->length() == 1;
            return true;
        }
    }
    return false;
}

const Def* World::extract(const Def* agg, const Def* index, Debug dbg) {
    if (agg->isa<Bottom>())
        return bottom(Extract::extracted_type(agg, index), dbg);
    if (agg->isa<Top>())
        return top(Extract::extracted_type(agg, index), dbg);

    if (!agg->isa<Param>() /*HACK*/ && fold_1_tuple(agg->type(), index))
        return agg;

    if (auto aggregate = agg->isa<Aggregate>()) {
        if (auto lit = index->isa<PrimLit>()) {
            if (!agg->isa<IndefiniteArray>()) {
                if (primlit_value<u64>(lit) < aggregate->num_ops())
                    return get(aggregate->ops(), lit);
                else
                    return bottom(Extract::extracted_type(agg, index), dbg);
            }
        }
    }

    if (auto insert = agg->isa<Insert>()) {
        if (index == insert->index())
            return insert->value();
        else if (index->template isa<PrimLit>()) {
            if (insert->index()->template isa<PrimLit>())
                return extract(insert->agg(), index, dbg);
        }
    }

    return unify(new Extract(agg, index, dbg));
}

const Def* World::insert(const Def* agg, const Def* index, const Def* value, Debug dbg) {
    if (agg->isa<Bottom>() || agg->isa<Top>()) {
        if (value->isa<Bottom>())
            return agg;

        // build aggregate container and fill with bottom
        if (auto definite_array_type = agg->type()->isa<DefiniteArrayType>()) {
            Array<const Def*> args(definite_array_type->dim());
            auto elem_type = definite_array_type->elem_type();
            auto elem = agg->isa<Bottom>() ? bottom(elem_type, dbg) : top(elem_type, dbg);
            std::fill(args.begin(), args.end(), elem);
            agg = definite_array(args, dbg);
        } else if (auto sigma = agg->type()->isa<Sigma>()) {
            Array<const Def*> args(sigma->num_ops());
            size_t i = 0;
            for (auto type : sigma->ops())
                args[i++] = agg->isa<Bottom>() ? bottom(type, dbg) : top(type, dbg);
            agg = tuple(sigma, args, dbg);
        }
    }

    if (fold_1_tuple(agg->type(), index))
        return value;

    // TODO double-check
    if (auto aggregate = agg->isa<Aggregate>()) {
        if (auto lit = index->isa<PrimLit>()) {
            if (!agg->isa<IndefiniteArray>()) {
                if (primlit_value<u64>(lit) < aggregate->num_ops()) {
                    Array<const Def*> args(agg->num_ops());
                    std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
                    args[primlit_value<u64>(lit)] = value;
                    return aggregate->rebuild(*this, agg->type(), args);
                } else
                    return bottom(agg->type(), dbg);
            }
        }
    }

    return unify(new Insert(agg, index, value, dbg));
}

const Def* World::lea(const Def* ptr, const Def* index, Debug dbg) {
    if (fold_1_tuple(ptr->type()->as<PtrType>()->pointee(), index))
        return ptr;

    return unify(new LEA(ptr, index, dbg));
}

const Def* World::select(const Def* cond, const Def* a, const Def* b, Debug dbg) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type(), dbg);

    if (auto lit = cond->isa<PrimLit>())
        return lit->value().get_bool() ? a : b;

    if (is_not(cond)) {
        cond = cond->as<ArithOp>()->rhs();
        std::swap(a, b);
    }

    if (a == b)
        return a;

    return unify(new Select(cond, a, b, dbg));
}

const Def* World::size_of(const Def* type, Debug dbg) {
    if (auto ptype = type->isa<PrimType>())
        return literal(qs32(num_bits(ptype->primtype_tag()) / 8), dbg);

    return unify(new SizeOf(bottom(type, dbg), dbg));
}

/*
 * memory stuff
 */

const Def* World::load(const Def* mem, const Def* ptr, Debug dbg) {
    if (auto sigma = ptr->type()->as<PtrType>()->pointee()->isa<Sigma>()) {
        // loading an empty tuple can only result in an empty tuple
        if (sigma->num_ops() == 0) {
            return tuple({mem, tuple({}, dbg)});
        }
    }
    return unify(new Load(mem, ptr, dbg));
}

bool is_agg_const(const Def* def) {
    return def->isa<DefiniteArray>() || def->isa<Tuple>();
}

const Def* World::store(const Def* mem, const Def* ptr, const Def* value, Debug dbg) {
    if (value->isa<Bottom>())
        return mem;
    return unify(new Store(mem, ptr, value, dbg));
}

const Def* World::enter(const Def* mem, Debug dbg) {
    if (auto e = Enter::is_out_mem(mem))
        return e;
    return unify(new Enter(mem, dbg));
}

const Def* World::alloc(const Def* type, const Def* mem, const Def* extra, Debug dbg) {
    return unify(new Alloc(type, mem, extra, dbg));
}

const Def* World::global(const Def* init, bool is_mutable, Debug dbg) {
    return unify(new Global(init, is_mutable, dbg));
}

const Def* World::global_immutable_string(const std::string& str, Debug dbg) {
    size_t size = str.size() + 1;

    Array<const Def*> str_array(size);
    for (size_t i = 0; i != size-1; ++i)
        str_array[i] = literal_qu8(str[i], dbg);
    str_array.back() = literal_qu8('\0', dbg);

    return global(definite_array(str_array, dbg), false, dbg);
}

const Assembly* World::assembly(const Def* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg) {
    return unify(new Assembly(type, inputs, asm_template, output_constraints, input_constraints, clobbers, flags, dbg))->as<Assembly>();;
}

const Assembly* World::assembly(Defs types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg) {
    Array<const Def*> output(types.size()+1);
    std::copy(types.begin(), types.end(), output.begin()+1);
    output.front() = mem_type();

    Array<const Def*> ops(inputs.size()+1);
    std::copy(inputs.begin(), inputs.end(), ops.begin()+1);
    ops.front() = mem;

    return assembly(sigma(output), ops, asm_template, output_constraints, input_constraints, clobbers, flags, dbg);
}

/*
 * partial evaluation related stuff
 */

const Def* World::hlt(const Def* def, Debug dbg) {
    if (pe_done_)
        return def;
    return unify(new Hlt(def, dbg));
}

const Def* World::known(const Def* def, Debug dbg) {
    if (pe_done_ || def->isa<Hlt>())
        return literal_bool(false, dbg);
    if (is_const(def))
        return literal_bool(true, dbg);
    return unify(new Known(def, dbg));
}

const Def* World::run(const Def* def, Debug dbg) {
    if (pe_done_)
        return def;
    return unify(new Run(def, dbg));
}

/*
 * lams
 */

Lam* World::match(const Def* type, size_t num_patterns) {
    Array<const Def*> arg_types(num_patterns + 2);
    arg_types[0] = type;
    arg_types[1] = cn();
    for (size_t i = 0; i < num_patterns; i++)
        arg_types[i + 2] = sigma({type, cn()});
    return lam(cn(sigma(arg_types)), CC::C, Intrinsic::Match, {"match"});
}

/*
 * misc
 */

const Def* World::try_fold_aggregate(const Aggregate* agg) {
    const Def* from = nullptr;
    for (size_t i = 0, e = agg->num_ops(); i != e; ++i) {
        auto arg = agg->op(i);
        if (auto extract = arg->isa<Extract>()) {
            if (from && extract->agg() != from) return agg;

            auto literal = extract->index()->isa<PrimLit>();
            if (!literal || literal->value().get_u64() != u64(i)) return agg;

            from = extract->agg();
        } else
            return agg;
    }
    return from && from->type() == agg->type() ? from : agg;
}

std::vector<Lam*> World::copy_lams() const {
    std::vector<Lam*> result;

    for (auto def : defs_) {
        if (auto lam = def->isa_lam())
            result.emplace_back(lam);
    }

    return result;
}

/*
 * optimizations
 */

void World::cleanup() { cleanup_world(*this); }

void World::opt() {
    cleanup();
    while (partial_evaluation(*this, true)); // lower2cff
    //flatten_tuples(*this);
    clone_bodies(*this);
    split_slots(*this);
    //closure_conversion(*this);
    lift_builtins(*this);
    inliner(*this);
    hoist_enters(*this);
    //dead_load_opt(*this);
    cleanup();
    codegen_prepare(*this);
    //rewrite_flow_graphs(*this);
}

/*
 * stream
 */

std::ostream& World::stream(std::ostream& os) const {
    os << "module '" << name() << "'\n\n";

    for (auto primop : defs()) {
        if (auto global = primop->isa<Global>())
            global->stream_assignment(os);
    }

    Scope::for_each<false>(*this, [&] (const Scope& scope) { scope.stream(os); });
    return os;
}

void World::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }

void World::thorin() const {
    auto filename = name() + ".thorin";
    write_thorin(filename.c_str());
}

}
