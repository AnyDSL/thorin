#include "thorin/world.h"

// for colored output
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if THORIN_ENABLE_CREATION_CONTEXT
#include <execinfo.h>
#endif

#if THORIN_ENABLE_RLIMITS
#include <sys/resource.h>
#endif

#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/continuation.h"
#include "thorin/type.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/plugin_execute.h"
#include "thorin/transform/closure_conversion.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/transform/dead_load_opt.h"
#include "thorin/transform/flatten_tuples.h"
#include "thorin/transform/hoist_enters.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/lift_builtins.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/transform/split_slots.h"
#include "thorin/util/array.h"

#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
#define THORIN_BREAK asm("int3");
#else
#define THORIN_BREAK { int* __p__ = nullptr; *__p__ = 42; }
#endif

namespace thorin {

/*
 * constructor and destructor
 */

World::World(const std::string& name) : types_(TypeTable(*this)) {
    data_.name_   = name;
    data_.branch_ = continuation(fn_type({mem_type(), type_bool(), fn_type({mem_type()}), fn_type({mem_type()})}), Intrinsic::Branch, {"br"});
    data_.end_scope_ = continuation(fn_type(), Intrinsic::EndScope, {"end_scope"});
}

const Def* World::variant_index(const Def* value, Debug dbg) {
    if (auto variant = value->isa<Variant>())
        return literal_qu64(variant->index(), dbg);
    return cse(new VariantIndex(*this, type_qu64(), value, dbg));
}

const Def* World::variant_extract(const Def* value, size_t index, Debug dbg) {
    auto type = value->type()->as<VariantType>()->op(index)->as<Type>();
    if (auto variant = value->isa<Variant>())
        return variant->index() == index ? variant->value() : bottom(type);
    return cse(new VariantExtract(*this, type, value, index, dbg));
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
#define THORIN_P_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())), dbg);
#define THORIN_Q_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())), dbg);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() && r.get_##T())), dbg);
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

    return cse(new ArithOp(tag, *this, a, b, dbg));
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

    return cse(new Cmp(tag, *this, a, b, dbg));
}

/*
 * casts
 */

const Def* World::convert(const Type* dst_type, const Def* src, Debug dbg) {
    if (dst_type == src->type())
        return src;
    if (src->type()->isa<PtrType>() && dst_type->isa<PtrType>())
        return bitcast(dst_type, src, dbg);
    if (auto dst_tuple_type = dst_type->isa<TupleType>()) {
        assert(dst_tuple_type->num_ops() == src->type()->as<TupleType>()->num_ops());

        Array<const Def*> new_tuple(dst_tuple_type->num_ops());
        for (size_t i = 0, e = new_tuple.size(); i != e; ++i)
            new_tuple[i] = convert(dst_tuple_type->types()[i], extract(src, i, dbg), dbg);

        return tuple(new_tuple, dbg);
    }

    return cast(dst_type, src, dbg);
}

const Def* World::cast(const Type* to, const Def* from, Debug dbg) {
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

    return cse(new Cast(*this, to, from, dbg));
}

const Def* World::bitcast(const Type* to, const Def* from, Debug dbg) {
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

    return cse(new Bitcast(*this, to, from, dbg));
}

/*
 * aggregate operations
 */

static bool fold_1_tuple(const Type* type, const Def* index) {
    if (auto lit = index->isa<PrimLit>()) {
        if (primlit_value<u64>(lit) == 0
                && !type->isa<ArrayType>()
                && !type->isa<StructType>()
                && !type->isa<TupleType>()) {
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

    if (fold_1_tuple(agg->type(), index))
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

    // TODO this doesn't work:
    // we have to use the current mem which is not necessarily ld->out_mem()
    //if (auto ld = Load::is_out_val(agg)) {
        //if (ld->out_val_type()->use_lea())
            //return extract(load(ld->out_mem(), lea(ld->ptr(), index, ld->name), name), 1);
    //}

    if (auto insert = agg->isa<Insert>()) {
        if (index == insert->index())
            return insert->value();
        else if (auto index_lit = index->isa<PrimLit>()) {
            if (auto insert_index_lit = insert->index()->isa<PrimLit>()) {
                if (index_lit->value() == insert_index_lit->value()) {
                    return insert->value();
                } else {
                    return extract(insert->agg(), index, dbg);
                }
            }
        }
    }

    return cse(new Extract(*this, agg, index, dbg));
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
        } else if (auto tuple_type = agg->type()->isa<TupleType>()) {
            Array<const Def*> args(tuple_type->num_ops());
            size_t i = 0;
            for (auto type : tuple_type->types())
                args[i++] = agg->isa<Bottom>() ? bottom(type, dbg) : top(type, dbg);
            agg = tuple(args, dbg);
        } else if (auto struct_type = agg->type()->isa<StructType>()) {
            Array<const Def*> args(struct_type->num_ops());
            size_t i = 0;
            for (auto type : struct_type->types())
                args[i++] = agg->isa<Bottom>() ? bottom(type, dbg) : top(type, dbg);
            agg = struct_agg(struct_type, args, dbg);
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
                    return aggregate->rebuild(*this, aggregate->type(), args);
                } else
                    return bottom(agg->type(), dbg);
            }
        }
    }

    return cse(new Insert(*this, agg, index, value, dbg));
}

const Def* World::lea(const Def* ptr, const Def* index, Debug dbg) {
    if (fold_1_tuple(ptr->type()->as<PtrType>()->pointee(), index))
        return ptr;

    return cse(new LEA(*this, ptr, index, dbg));
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

    return cse(new Select(*this, cond, a, b, dbg));
}

const Def* World::align_of(const Type* type, Debug dbg) {
    if (auto ptype = type->isa<PrimType>())
        return literal(qs64(num_bits(ptype->primtype_tag()) / 8), dbg);

    return cse(new AlignOf(*this, bottom(type, dbg), dbg));
}

const Def* World::size_of(const Type* type, Debug dbg) {
    if (auto ptype = type->isa<PrimType>())
        return literal(qs64(num_bits(ptype->primtype_tag()) / 8), dbg);

    return cse(new SizeOf(*this, bottom(type, dbg), dbg));
}

/*
 * mathematical operations
 */

template <class F>
const Def* World::transcendental(MathOpTag tag, const Def* arg, Debug dbg, F&& f) {
    assert(is_type_f(arg->type()));
    if (auto lit = arg->isa<PrimLit>()) {
        switch (lit->primtype_tag()) {
            case PrimType_qf16:
            case PrimType_pf16:
                return literal(lit->primtype_tag(), Box(f(lit->value().get_f16())), dbg);
            case PrimType_qf32:
            case PrimType_pf32:
                return literal(lit->primtype_tag(), Box(f(lit->value().get_f32())), dbg);
            case PrimType_qf64:
            case PrimType_pf64:
                return literal(lit->primtype_tag(), Box(f(lit->value().get_f64())), dbg);
            default:
                THORIN_UNREACHABLE;
        }
    }
    return cse(new MathOp(*this, tag, arg->type(), { arg }, dbg));
}

template <class F>
const Def* World::transcendental(MathOpTag tag, const Def* left, const Def* right, Debug dbg, F&& f) {
    assert(left->type() == right->type());
    assert(is_type_f(left->type()));
    if (auto [left_lit, right_lit] = std::pair { left->isa<PrimLit>(), right->isa<PrimLit>() }; left_lit && right_lit) {
        switch (left_lit->primtype_tag()) {
            case PrimType_qf16:
            case PrimType_pf16:
                return literal(left_lit->primtype_tag(), Box(f(left_lit->value().get_f16(), right_lit->value().get_f16())), dbg);
            case PrimType_qf32:
            case PrimType_pf32:
                return literal(left_lit->primtype_tag(), Box(f(left_lit->value().get_f32(), right_lit->value().get_f32())), dbg);
            case PrimType_qf64:
            case PrimType_pf64:
                return literal(left_lit->primtype_tag(), Box(f(left_lit->value().get_f64(), right_lit->value().get_f64())), dbg);
            default:
                THORIN_UNREACHABLE;
        }
    }
    return cse(new MathOp(*this, tag, left->type(), { left, right }, dbg));
}

template <class F>
static inline bool float_predicate(const PrimLit* lit, F&& f) {
    switch (lit->primtype_tag()) {
        case PrimType_qf16:
        case PrimType_pf16:
            return f(lit->value().get_f16());
        case PrimType_qf32:
        case PrimType_pf32:
            return f(lit->value().get_f32());
        case PrimType_qf64:
        case PrimType_pf64:
            return f(lit->value().get_f64());
        default:
            THORIN_UNREACHABLE;
    }
}

const Def* World::mathop(MathOpTag tag, Defs args, Debug dbg) {
    // Folding rules are only valid for fast-math floating-point types
    // No attempt to simplify mathematical expressions will be attempted otherwise
    auto signbit = [] (auto x) {
        using T = decltype(x);
        if constexpr (std::is_same_v<T, half>) return half_float::signbit(x);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return std::signbit(x);
        THORIN_UNREACHABLE;
    };
    if (tag == MathOp_copysign) {
        if (is_type_qf(args[1]->type()) && args[1]->isa<PrimLit>()) {
            // - copysign(x, <known-constant>) => -x if signbit(<known_constant>) or x otherwise
            return float_predicate(args[1]->as<PrimLit>(), signbit) ? arithop_minus(args[0], dbg) : args[0];
        }
        return transcendental(MathOp_copysign, args[0], args[1], dbg, [] (auto x, auto y) -> decltype(x) {
            using T = decltype(x);
            if constexpr (std::is_same_v<T, half>) return half_float::copysign(x, y);
            else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return std::copysign(x, y);
            THORIN_UNREACHABLE;
        });
    } else if (tag == MathOp_pow) {
        if (is_type_qf(args[1]->type()) &&
            args[1]->isa<PrimLit>() &&
            float_predicate(args[1]->as<PrimLit>(), [] (auto x) { return x == decltype(x)(0.5); }))
        {
            // - pow(x, 0.5) => sqrt(x)
            return sqrt(args[0], dbg);
        }
        return transcendental(MathOp_pow, args[0], args[1], dbg, [] (auto x, auto y) -> decltype(x) {
            using T = decltype(x);
            if constexpr (std::is_same_v<T, half>) return half_float::pow(x, y);
            else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return std::pow(x, y);
            THORIN_UNREACHABLE;
        });
    } else if (tag == MathOp_atan2) {
        return transcendental(MathOp_atan2, args[0], args[1], dbg, [] (auto x, auto y) -> decltype(x) {
            using T = decltype(x);
            if constexpr (std::is_same_v<T, half>) return half_float::atan2(x, y);
            else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return std::atan2(x, y);
            THORIN_UNREACHABLE;
        });
    } else if (tag == MathOp_fmin) {
        return transcendental(MathOp_fmin, args[0], args[1], dbg, [] (auto x, auto y) -> decltype(x) {
            using T = decltype(x);
            if constexpr (std::is_same_v<T, half>) return half_float::fmin(x, y);
            else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return std::fmin(x, y);
            THORIN_UNREACHABLE;
        });
    } else if (tag == MathOp_fmax) {
        return transcendental(MathOp_fmax, args[0], args[1], dbg, [] (auto x, auto y) -> decltype(x) {
            using T = decltype(x);
            if constexpr (std::is_same_v<T, half>) return half_float::fmax(x, y);
            else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return std::fmax(x, y);
            THORIN_UNREACHABLE;
        });
    } else {
        if (is_type_qf(args[0]->type())) {
            // - cos(acos(x)) => x
            // - sin(asin(x)) => x
            // - tan(atan(x)) => x
            // Note: The other way around (i.e. `acos(cos(x)) => x`) is not always true
            if (args[0]->isa<MathOp>()) {
                auto other_tag = args[0]->as<MathOp>()->mathop_tag();
                switch (tag) {
                    case MathOp_cos: if (other_tag == MathOp_acos) return args[0]->op(0); break;
                    case MathOp_sin: if (other_tag == MathOp_asin) return args[0]->op(0); break;
                    case MathOp_tan: if (other_tag == MathOp_atan) return args[0]->op(0); break;
                    default: break;
                }
            }
            // - sqrt(x * x) => x
            // - cbrt(x * x * x) => x
            if (args[0]->isa<ArithOp>()) {
                auto is_square = [] (const Def* x) -> const Def* {
                    if (auto arithop = x->isa<ArithOp>(); arithop && arithop->arithop_tag() == ArithOp_mul)
                        return x->op(0) == x->op(1) ? x->op(0) : nullptr;
                    return nullptr;
                };
                auto is_cube = [&] (const Def* x) -> const Def* {
                    if (auto arithop = x->isa<ArithOp>(); arithop && arithop->arithop_tag() == ArithOp_mul) {
                        if (auto lhs = is_square(arithop->op(0)); lhs == arithop->op(1)) return lhs;
                        if (auto rhs = is_square(arithop->op(1)); rhs == arithop->op(0)) return rhs;
                    }
                    return nullptr;
                };
                switch (tag) {
                    case MathOp_sqrt: if (auto x = is_square(args[0])) return x; break;
                    case MathOp_cbrt: if (auto x = is_cube(args[0]))   return x; break;
                    default: break;
                }
            }
        }
        return transcendental(tag, args[0], dbg, [&] (auto arg) -> decltype(arg) {
            using T = decltype(arg);
            if constexpr (std::is_same_v<T, half>) {
                switch (tag) {
                    case MathOp_fabs:  return half_float::fabs(arg);
                    case MathOp_round: return half_float::round(arg);
                    case MathOp_floor: return half_float::floor(arg);
                    case MathOp_ceil:  return half_float::ceil(arg);
                    case MathOp_cos:   return half_float::cos(arg);
                    case MathOp_sin:   return half_float::sin(arg);
                    case MathOp_tan:   return half_float::tan(arg);
                    case MathOp_acos:  return half_float::acos(arg);
                    case MathOp_asin:  return half_float::asin(arg);
                    case MathOp_atan:  return half_float::atan(arg);
                    case MathOp_sqrt:  return half_float::sqrt(arg);
                    case MathOp_cbrt:  return half_float::cbrt(arg);
                    case MathOp_exp:   return half_float::exp(arg);
                    case MathOp_exp2:  return half_float::exp2(arg);
                    case MathOp_log:   return half_float::log(arg);
                    case MathOp_log2:  return half_float::log2(arg);
                    case MathOp_log10: return half_float::log10(arg);
                    default: break;
                }
            } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                switch (tag) {
                    case MathOp_fabs:  return std::fabs(arg);
                    case MathOp_round: return std::round(arg);
                    case MathOp_floor: return std::floor(arg);
                    case MathOp_ceil:  return std::ceil(arg);
                    case MathOp_cos:   return std::cos(arg);
                    case MathOp_sin:   return std::sin(arg);
                    case MathOp_tan:   return std::tan(arg);
                    case MathOp_acos:  return std::acos(arg);
                    case MathOp_asin:  return std::asin(arg);
                    case MathOp_atan:  return std::atan(arg);
                    case MathOp_sqrt:  return std::sqrt(arg);
                    case MathOp_cbrt:  return std::cbrt(arg);
                    case MathOp_exp:   return std::exp(arg);
                    case MathOp_exp2:  return std::exp2(arg);
                    case MathOp_log:   return std::log(arg);
                    case MathOp_log2:  return std::log2(arg);
                    case MathOp_log10: return std::log10(arg);
                    default: break;
                }
            }
            THORIN_UNREACHABLE;
        });
    }
}

/*
 * memory stuff
 */

const Def* World::load(const Def* mem, const Def* ptr, Debug dbg) {
    if (auto tuple_type = ptr->type()->as<PtrType>()->pointee()->isa<TupleType>()) {
        // loading an empty tuple can only result in an empty tuple
        if (tuple_type->num_ops() == 0) {
            return tuple({mem, tuple({}, dbg)});
        }
    }
    return cse(new Load(*this, mem, ptr, dbg));
}

bool is_agg_const(const Def* def) {
    return def->isa<DefiniteArray>() || def->isa<StructAgg>() || def->isa<Tuple>();
}

const Def* World::store(const Def* mem, const Def* ptr, const Def* value, Debug dbg) {
    if (value->isa<Bottom>())
        return mem;
    return cse(new Store(*this, mem, ptr, value, dbg));
}

const Def* World::enter(const Def* mem, Debug dbg) {
    // while hoist_enters performs this task too, this is still necessary for PE
    // in order to simplify as we go and prevent code size from exploding
    if (auto e = Enter::is_out_mem(mem))
        return e;
    return cse(new Enter(*this, mem, dbg));
}

const Def* World::alloc(const Type* type, const Def* mem, const Def* extra, Debug dbg) {
    return cse(new Alloc(*this, type, mem, extra, dbg));
}

const Def* World::release(const Def* mem, const Def* alloc, Debug dbg) {
    return cse(new Release(*this, mem, alloc, dbg));
}

const Def* World::global(const Def* init, bool is_mutable, Debug dbg) {
    return cse(new Global(*this, init, is_mutable, dbg));
}

const Def* World::global_immutable_string(const std::string& str, Debug dbg) {
    size_t size = str.size() + 1;

    Array<const Def*> str_array(size);
    for (size_t i = 0; i != size-1; ++i)
        str_array[i] = literal_qu8(str[i], dbg);
    str_array.back() = literal_qu8('\0', dbg);

    return global(definite_array(str_array, dbg), false, dbg);
}

const Assembly* World::assembly(const Type* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg) {
    return cse(new Assembly(*this, type, inputs, asm_template, output_constraints, input_constraints, clobbers, flags, dbg))->as<Assembly>();;
}

const Assembly* World::assembly(Types types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg) {
    Array<const Type*> output(types.size()+1);
    std::copy(types.begin(), types.end(), output.begin()+1);
    output.front() = mem_type();

    Array<const Def*> ops(inputs.size()+1);
    std::copy(inputs.begin(), inputs.end(), ops.begin()+1);
    ops.front() = mem;

    return assembly(tuple_type(output), ops, asm_template, output_constraints, input_constraints, clobbers, flags, dbg);
}

/*
 * partial evaluation related stuff
 */

const Def* World::hlt(const Def* def, Debug dbg) {
    if (is_pe_done()) return def;
    return cse(new Hlt(*this, def, dbg));
}

const Def* World::known(const Def* def, Debug dbg) {
    if (is_pe_done() || def->isa<Hlt>())
        return literal_bool(false, dbg);
    if (!def->has_dep(Dep::Param))
        return literal_bool(true, dbg);
    return cse(new Known(*this, def, dbg));
}

const Def* World::run(const Def* def, Debug dbg) {
    if (is_pe_done()) return def;
    return cse(new Run(*this, def, dbg));
}

/*
 * continuations
 */

Continuation* World::continuation(const FnType* fn, Continuation::Attributes attributes, Debug dbg) {
#if THORIN_ENABLE_CREATION_CONTEXT
    void *array[10];
    size_t size = backtrace(array, 10);
    assert(size >= 2);
    char ** symbols = backtrace_symbols(array, 10);

    dbg.creation_context = symbols[1];
#endif

    auto cont = put<Continuation>(*this, fn, attributes, dbg);

#if THORIN_ENABLE_CREATION_CONTEXT
    free(symbols);
#endif

    return cont;
}

Continuation* World::match(const Type* type, size_t num_patterns) {
    Array<const Type*> arg_types(num_patterns + 3);
    arg_types[0] = mem_type();
    arg_types[1] = type;
    arg_types[2] = fn_type({mem_type()});
    for (size_t i = 0; i < num_patterns; i++) {
        arg_types[i + 3] = tuple_type({type, fn_type({mem_type()})});
    }
    return continuation(fn_type(arg_types), Intrinsic::Match, {"match"});
}

const Param* World::param(const Type* type, const Continuation* continuation, size_t index, Debug dbg) {
    auto param = cse(new Param(*this, type, continuation, index, dbg));
#if THORIN_ENABLE_CHECKS
    if (state_.breakpoints.contains(param->gid())) THORIN_BREAK;
#endif
    return param;
}

const Filter* World::filter(const Defs defs, Debug dbg) {
    return cse(new Filter(*this, defs, dbg));
}

/// App node does its own folding during construction, and it only sets the ops once
const App* World::app(const Def* callee, const Defs args, Debug dbg) {
    if (auto continuation = callee->isa<Continuation>()) {
        switch (continuation->intrinsic()) {
            // See also mangle::instantiate when modifying this.
            case Intrinsic::Branch: {
                assert(args.size() == 4);
                auto mem = args[0], cond = args[1], t = args[2], f = args[3];
                if (auto lit = cond->isa<PrimLit>())
                    return app(lit->value().get_bool() ? t : f, { mem }, dbg);
                if (t == f)
                    return app(t, { mem }, dbg);
                if (is_not(cond)) {
                    auto inverted = cond->as<ArithOp>()->rhs();
                    return app(branch(), {mem, inverted, f, t}, dbg);
                }
                break;
            }
            case Intrinsic::Match:
                if (args.size() == 3) return app(args[2], { args[0] }, dbg);
                if (auto lit = args[1]->isa<PrimLit>()) {
                    for (size_t i = 3; i < args.size(); i++) {
                        if (extract(args[i], 0_s)->as<PrimLit>() == lit)
                            return app(extract(args[i], 1), { args[0] }, dbg);
                    }
                    return app(args[2], { args[0] }, dbg);
                }
                break;
            default:
                break;
        }
    }

    Array<const Def*> ops(1 + args.size());
    ops[0] = callee;
    for (size_t i = 0; i < args.size(); i++)
        ops[i + 1] = args[i];

    return cse(new App(*this, ops, dbg));
}

/*
 * misc
 */

std::vector<Continuation*> World::copy_continuations() const {
    std::vector<Continuation*> result;

    for (auto def : data_.defs_) {
        if (auto lam = def->isa_nom<Continuation>())
            result.emplace_back(lam);
    }

    return result;
}
#if THORIN_ENABLE_CHECKS

void World::    breakpoint(size_t number) { state_.    breakpoints.insert(number); }
void World::use_breakpoint(size_t number) { state_.use_breakpoints.insert(number); }
void World::enable_history(bool flag)     { state_.track_history = flag; }
bool World::track_history() const         { return state_.track_history; }

const Def* World::gid2def(u32 gid) {
    auto i = std::find_if(data_.defs_.begin(), data_.defs_.end(), [&](const Def* def) { return def->gid() == gid; });
    if (i == data_.defs_.end()) return nullptr;
    return *i;
}

#endif

const char* World::level2string(LogLevel level) {
    switch (level) {
        case LogLevel::Error:   return "E";
        case LogLevel::Warn:    return "W";
        case LogLevel::Info:    return "I";
        case LogLevel::Verbose: return "V";
        case LogLevel::Debug:   return "D";
    }
    THORIN_UNREACHABLE;
}

int World::level2color(LogLevel level) {
    switch (level) {
        case LogLevel::Error:   return 1;
        case LogLevel::Warn:    return 3;
        case LogLevel::Info:    return 2;
        case LogLevel::Verbose: return 4;
        case LogLevel::Debug:   return 4;
    }
    THORIN_UNREACHABLE;
}

#ifdef COLORIZE_LOG
std::string World::colorize(const std::string& str, int color) {
    if (isatty(fileno(stdout))) {
        const char c = '0' + color;
        return "\033[1;3" + (c + ('m' + str)) + "\033[0m";
    }
#else
std::string World::colorize(const std::string& str, int) {
#endif
    return str;
}

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

const Def* World::cse_base(const Def* def) {
    assert(def->isa_structural());
#if THORIN_ENABLE_CHECKS
    if (state_.breakpoints.contains(def->gid())) THORIN_BREAK;
#endif
    auto i = data_.defs_.find(def);
    if (i != data_.defs_.end()) {
        def->unregister_uses();
        --Def::gid_counter_;
        delete def;
        return *i;
    }

    const auto& p = data_.defs_.insert(def);
    assert_unused(p.second && "hash/equal broken");
    return def;
}

/*
 * optimizations
 */

Thorin::Thorin(const std::string& name)
    : world_(std::make_unique<World>(name))
{}

void Thorin::opt() {
    bool debug_passes = getenv("THORIN_DEBUG_PASSES");
#define RUN_PASS(pass) \
{ \
    world().VLOG("running pass {}", #pass);  \
    pass;                                    \
    debug_verify(world());                   \
    if (debug_passes) world().dump_scoped(); \
}

    RUN_PASS(cleanup())
    RUN_PASS(while (partial_evaluation(world(), true))); // lower2cff
    RUN_PASS(flatten_tuples(*this))
    RUN_PASS(split_slots(*this))
    if (plugin_handles.size() > 0) {
        RUN_PASS(plugin_execute(*this));
        RUN_PASS(cleanup());
    }
    RUN_PASS(closure_conversion(world()))
    RUN_PASS(lift_builtins(*this))
    RUN_PASS(inliner(*this))
    RUN_PASS(hoist_enters(*this))
    RUN_PASS(dead_load_opt(world()))
    RUN_PASS(cleanup())
    RUN_PASS(codegen_prepare(*this))
}

bool Thorin::ensure_stack_size(size_t new_size) {
#if THORIN_ENABLE_RLIMITS
    struct rlimit rl;
    int result = getrlimit(RLIMIT_STACK, &rl);
    if(result != 0) return false;

    rl.rlim_cur = new_size;
    result = setrlimit(RLIMIT_STACK, &rl);
    if(result != 0) return false;

    return true;
#else
    return false;
#endif
}

bool Thorin::register_plugin(const char* plugin_name) {
#ifdef _MSC_VER
    return false;
#else // _MSC_VER
    void *handle = dlopen(plugin_name, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        world().ELOG("Error loading plugin {}: {}", plugin_name, dlerror());
        world().ELOG("Is plugin contained in LD_LIBRARY_PATH?");
        return false;
    }
    dlerror();

    char *error;
    auto initfunc = reinterpret_cast<plugin_init_func_t*>(dlsym(handle, "init"));
    if ((error = dlerror()) != NULL) {
        world().ILOG("Plugin {} did not provide an init function", plugin_name);
    } else {
        initfunc(&world());
    }

    plugin_handles.push_back(handle);
    return true;
#endif // _MSC_VER
}

Thorin::plugin_func_t* Thorin::search_plugin_function(const char* function_name) const {
#ifdef _MSC_VER
#else // _MSC_VER
    for (auto plugin : plugin_handles) {
        if (void* plugin_function = dlsym(plugin, function_name)) {
            return reinterpret_cast<plugin_func_t*>(plugin_function);
        }
    }
#endif // _MSC_VER
    return nullptr;
}
}
