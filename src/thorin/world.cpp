#include "thorin/world.h"

#include <fstream>

#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/continuation.h"
#include "thorin/type.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/clone_bodies.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/lift_builtins.h"
#include "thorin/transform/lift_enters.h"
#include "thorin/transform/lower2cff.h"
#include "thorin/transform/mem2reg.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/transform/dead_load_opt.h"
#include "thorin/util/array.h"
#include "thorin/util/log.h"

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
    , fn0_    (unify(new FnType   (*this, Types(), 0)))
    , mem_    (unify(new MemType  (*this)))
    , frame_  (unify(new FrameType(*this)))
#define THORIN_ALL_TYPE(T, M) \
    ,T##_(unify(new PrimType(*this, PrimType_##T, 1)))
#include "thorin/tables/primtypetable.h"
{
    branch_ = continuation(fn_type({type_bool(), fn_type(), fn_type()}), Location(), CC::C, Intrinsic::Branch, "br");
    end_scope_ = continuation(fn_type(), Location(), CC::C, Intrinsic::EndScope, "end_scope");
}

World::~World() {
    for (auto continuation : continuations_) delete continuation;
    for (auto primop : primops_) delete primop;
}

const StructAbsType* World::struct_abs_type(size_t size, size_t num_type_params, const std::string& name) {
    auto struct_abs_type = new StructAbsType(*this, size, num_type_params, name);
    // just put it into the types_ set due to nominal typing
    auto p = types_.insert(struct_abs_type);
    assert_unused(p.second && "hash/equal broken");
    struct_abs_type->hashed_ = true;
    return struct_abs_type;
}

/*
 * literals
 */

const Def* World::splat(const Def* arg, size_t length, const std::string& name) {
    if (length == 1)
        return arg;

    Array<const Def*> args(length);
    std::fill(args.begin(), args.end(), arg);
    return vector(args, arg->loc(), name);
}

/*
 * arithops
 */

const Def* World::binop(int kind, const Def* lhs, const Def* rhs, const Location& loc, const std::string& name) {
    if (is_arithop(kind))
        return arithop((ArithOpKind) kind, lhs, rhs, loc, name);

    assert(is_cmp(kind) && "must be a Cmp");
    return cmp((CmpKind) kind, lhs, rhs, loc, name);
}

const Def* World::arithop(ArithOpKind kind, const Def* a, const Def* b, const Location& loc, const std::string& name) {
    assert(a->type() == b->type());
    assert(a->type()->as<PrimType>()->length() == b->type()->as<PrimType>()->length());
    PrimTypeKind type = a->type()->as<PrimType>()->primtype_kind();

    auto llit = a->isa<PrimLit>();
    auto rlit = b->isa<PrimLit>();
    auto lvec = a->isa<Vector>();
    auto rvec = b->isa<Vector>();

    if (lvec && rvec) {
        size_t num = lvec->type()->as<PrimType>()->length();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = arithop(kind, lvec->op(i), rvec->op(i), loc);
        return vector(ops, loc, name);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();

        try {
            switch (kind) {
                case ArithOp_add:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_sub:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_mul:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_div:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_rem:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() % r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                    }
                case ArithOp_and:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())), loc);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() & r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_or:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())), loc);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() | r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_xor:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())), loc);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() ^ r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shl:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() << r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shr:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return literal(type, Box(T(l.get_##T() >> r.get_##T())), loc);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
            }
        } catch (BottomException) {
            return bottom(type, loc);
        }
    }

    // normalize: swap literal/vector to the left
    if (is_commutative(kind) && (rlit || rvec)) {
        std::swap(a, b);
        std::swap(llit, rlit);
        std::swap(lvec, rvec);
    }

    if (is_type_i(type)) {
        if (a == b) {
            switch (kind) {
                case ArithOp_add: return arithop_mul(literal(type, 2, loc), a, loc);

                case ArithOp_sub:
                case ArithOp_xor: return zero(type, loc);

                case ArithOp_and:
                case ArithOp_or:  return a;

                case ArithOp_div:
                    if (b->is_zero())
                        return bottom(type, loc);
                    return one(type, loc);

                case ArithOp_rem:
                    if (b->is_zero())
                        return bottom(type, loc);
                    return zero(type, loc);

                default: break;
            }
        }

        if (a->is_zero()) {
            switch (kind) {
                case ArithOp_mul:
                case ArithOp_div:
                case ArithOp_rem:
                case ArithOp_and:
                case ArithOp_shl:
                case ArithOp_shr: return zero(type, loc);

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

        if (b->is_zero()) {
            switch (kind) {
                case ArithOp_div:
                case ArithOp_rem: return bottom(type, loc);

                case ArithOp_shl:
                case ArithOp_shr: return a;

                default: break;
            }
        }

        if (b->is_one()) {
            switch (kind) {
                case ArithOp_mul:
                case ArithOp_div: return a;
                case ArithOp_rem: return zero(type, loc);

                default: break;
            }
        }

        if (rlit && primlit_value<uint64_t>(rlit) >= uint64_t(num_bits(type))) {
            switch (kind) {
                case ArithOp_shl:
                case ArithOp_shr: return bottom(type, loc);

                default: break;
            }
        }

        if (kind == ArithOp_xor && a->is_allset()) {    // is this a NOT
            if (b->is_not())                            // do we have ~~x?
                return b->as<ArithOp>()->rhs();
            if (auto cmp = b->isa<Cmp>())   // do we have ~(a cmp b)?
                return this->cmp(negate(cmp->cmp_kind()), cmp->lhs(), cmp->rhs(), loc);
        }

        auto lcmp = a->isa<Cmp>();
        auto rcmp = b->isa<Cmp>();

        if (kind == ArithOp_or && lcmp && rcmp && lcmp->lhs() == rcmp->lhs() && lcmp->rhs() == rcmp->rhs()
                && lcmp->cmp_kind() == negate(rcmp->cmp_kind()))
                return literal_bool(true, loc);

        if (kind == ArithOp_and && lcmp && rcmp && lcmp->lhs() == rcmp->lhs() && lcmp->rhs() == rcmp->rhs()
                && lcmp->cmp_kind() == negate(rcmp->cmp_kind()))
                return literal_bool(false, loc);

        auto land = a->kind() == Node_and ? a->as<ArithOp>() : nullptr;
        auto rand = b->kind() == Node_and ? b->as<ArithOp>() : nullptr;

        // distributivity (a and b) or (a and c)
        if (kind == ArithOp_or && land && rand) {
            if (land->lhs() == rand->lhs())
                return arithop_and(land->lhs(), arithop_or(land->rhs(), rand->rhs(), loc), loc);
            if (land->rhs() == rand->rhs())
                return arithop_and(land->rhs(), arithop_or(land->lhs(), rand->lhs(), loc), loc);
        }

        auto lor = a->kind() == Node_or ? a->as<ArithOp>() : nullptr;
        auto ror = b->kind() == Node_or ? b->as<ArithOp>() : nullptr;

        // distributivity (a or b) and (a or c)
        if (kind == ArithOp_and && lor && ror) {
            if (lor->lhs() == ror->lhs())
                return arithop_or(lor->lhs(), arithop_and(lor->rhs(), ror->rhs(), loc), loc);
            if (lor->rhs() == ror->rhs())
                return arithop_or(lor->rhs(), arithop_and(lor->lhs(), ror->lhs(), loc), loc);
        }

        // absorption: a and (a or b) = a
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

        // absorption: a or (a and b) = a
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
                    return arithop_or(lor->rhs(), ror->rhs(), loc);
                if (lor->rhs() == ror->rhs())
                    return arithop_or(lor->lhs(), ror->lhs(), loc);
            }
        }

        if (kind == ArithOp_and) {
            if (land && rand) {
                if (land->lhs() == rand->lhs())
                    return arithop_and(land->rhs(), rand->rhs(), loc);
                if (land->rhs() == rand->rhs())
                    return arithop_and(land->lhs(), rand->lhs(), loc);
            }
        }
    }

    // normalize: try to reorder same ops to have the literal/vector on the left-most side
    if (is_associative(kind) && is_type_i(a->type())) {
        auto a_same = a->isa<ArithOp>() && a->as<ArithOp>()->arithop_kind() == kind ? a->as<ArithOp>() : nullptr;
        auto b_same = b->isa<ArithOp>() && b->as<ArithOp>()->arithop_kind() == kind ? b->as<ArithOp>() : nullptr;
        auto a_lhs_lv = a_same && (a_same->lhs()->isa<PrimLit>() || a_same->lhs()->isa<Vector>()) ? a_same->lhs() : nullptr;
        auto b_lhs_lv = b_same && (b_same->lhs()->isa<PrimLit>() || b_same->lhs()->isa<Vector>()) ? b_same->lhs() : nullptr;

        if (is_commutative(kind)) {
            if (a_lhs_lv && b_lhs_lv)
                return arithop(kind, arithop(kind, a_lhs_lv, b_lhs_lv, loc), arithop(kind, a_same->rhs(), b_same->rhs(), loc), loc);
            if ((llit || lvec) && b_lhs_lv)
                return arithop(kind, arithop(kind, a, b_lhs_lv, loc), b_same->rhs(), loc);
            if (b_lhs_lv)
                return arithop(kind, b_lhs_lv, arithop(kind, a, b_same->rhs(), loc), loc);
        }
        if (a_lhs_lv)
            return arithop(kind, a_lhs_lv, arithop(kind, a_same->rhs(), b, loc), loc);
    }

    return cse(new ArithOp(kind, a, b, loc, name));
}

const Def* World::arithop_not(const Def* def, const Location& loc) { return arithop_xor(allset(def->type(), loc, def->length()), def, loc); }

const Def* World::arithop_minus(const Def* def, const Location& loc) {
    switch (PrimTypeKind kind = def->type()->as<PrimType>()->primtype_kind()) {
#define THORIN_F_TYPE(T, M) \
        case PrimType_##T: \
            return arithop_sub(literal_##T(M(-0.f), loc, def->length()), def, loc);
#include "thorin/tables/primtypetable.h"
        default:
            assert(is_type_i(kind));
            return arithop_sub(zero(kind, loc), def, loc);
    }
}

/*
 * compares
 */

const Def* World::cmp(CmpKind kind, const Def* a, const Def* b, const Location& loc, const std::string& name) {
    CmpKind oldkind = kind;
    switch (kind) {
        case Cmp_gt:  kind = Cmp_lt; break;
        case Cmp_ge:  kind = Cmp_le; break;
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
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cmp(kind, lvec->op(i), rvec->op(i), loc);
        return vector(ops, loc, name);
    }

    if (llit && rlit) {
        Box l = llit->value();
        Box r = rlit->value();
        PrimTypeKind type = llit->primtype_kind();

        // TODO unordered
        switch (kind) {
            case Cmp_eq:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() == r.get_##T(), loc);
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_ne:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() != r.get_##T(), loc);
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_lt:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() <  r.get_##T(), loc);
#include "thorin/tables/primtypetable.h"
                }
            case Cmp_le:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_bool(l.get_##T() <= r.get_##T(), loc);
#include "thorin/tables/primtypetable.h"
                }
            default: THORIN_UNREACHABLE;
        }
    }

    if (a == b) {
        switch (kind) {
            case Cmp_lt:
            case Cmp_ne:  return zero(type_bool(), loc);
            case Cmp_le:
            case Cmp_eq:  return one(type_bool(), loc);
            default: break;
        }
    }

    return cse(new Cmp(kind, a, b, loc, name));
}

/*
 * casts
 */

const Def* World::convert(const Type* to, const Def* from, const Location& loc, const std::string& name) {
    if (from->type()->isa<PtrType>() && to->isa<PtrType>())
        return bitcast(to, from, loc, name);
    return cast(to, from, loc, name);
}

const Def* World::cast(const Type* to, const Def* from, const Location& loc, const std::string& name) {
    if (auto vec = from->isa<Vector>()) {
        size_t num = vec->length();
        auto to_vec = to->as<VectorType>();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = cast(to_vec->scalarize(), vec->op(i), loc);
        return vector(ops, loc, name);
    }

    auto lit = from->isa<PrimLit>();
    auto to_type = to->isa<PrimType>();
    if (lit && to_type) {
        Box box = lit->value();

        switch (lit->primtype_kind()) {
            case PrimType_bool:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_bool()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps8:
            case PrimType_qs8:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s8()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps16:
            case PrimType_qs16:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s16()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps32:
            case PrimType_qs32:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s32()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_ps64:
            case PrimType_qs64:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_s64()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu8:
            case PrimType_qu8:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u8()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu16:
            case PrimType_qu16:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u16()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu32:
            case PrimType_qu32:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u32()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pu64:
            case PrimType_qu64:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_u64()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pf16:
            case PrimType_qf16:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_f16()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pf32:
            case PrimType_qf32:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_f32()), loc);
#include "thorin/tables/primtypetable.h"
                }
            case PrimType_pf64:
            case PrimType_qf64:
                switch (to_type->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return literal_##T(M(box.get_f64()), loc);
#include "thorin/tables/primtypetable.h"
                }
        }
    }

    return cse(new Cast(to, from, loc, name));
}

const Def* World::bitcast(const Type* to, const Def* from, const Location& loc, const std::string& name) {
    if (auto other = from->isa<Bitcast>()) {
        if (to == other->type())
            return other;
    }

    if (auto vec = from->isa<Vector>()) {
        size_t num = vec->length();
        auto to_vec = to->as<VectorType>();
        Array<const Def*> ops(num);
        for (size_t i = 0; i != num; ++i)
            ops[i] = bitcast(to_vec->scalarize(), vec->op(i), loc);
        return vector(ops, loc, name);
    }

    // TODO constant folding
    return cse(new Bitcast(to, from, loc, name));
}

/*
 * aggregate operations
 */

template<class T>
const Def* World::extract(const Def* agg, const T* index, const Location& loc, const std::string& name) {
    if (agg->isa<Bottom>())
        return bottom(Extract::extracted_type(agg, index), loc);

    if (auto aggregate = agg->isa<Aggregate>()) {
        if (auto lit = index->template isa<PrimLit>()) {
            if (!agg->isa<IndefiniteArray>())
                return get(aggregate->ops(), lit);
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
        else if (index->template isa<PrimLit>()) {
            if (insert->index()->template isa<PrimLit>())
                return extract(insert->agg(), index, loc, name);
        }
    }

    return cse(new Extract(agg, index, loc, name));
}

// this workaround is need in order to disambiguate extract(agg, 0, ...) from
// extract(..., u32, ...) vs extract(..., const Def*, ...)
template const Def* World::extract<Def>    (const Def*, const Def*,     const Location&, const std::string&); // instantiate
template const Def* World::extract<PrimLit>(const Def*, const PrimLit*, const Location&, const std::string&); // methods

const Def* World::insert(const Def* agg, const Def* index, const Def* value, const Location& loc, const std::string& name) {
    if (agg->isa<Bottom>()) {
        if (value->isa<Bottom>())
            return agg;

        // build aggregate container and fill with bottom
        if (auto definite_array_type = agg->type()->isa<DefiniteArrayType>()) {
            Array<const Def*> args(definite_array_type->dim());
            std::fill(args.begin(), args.end(), bottom(definite_array_type->elem_type(), loc));
            agg = definite_array(args, loc, agg->name);
        } else if (auto tuple_type = agg->type()->isa<TupleType>()) {
            Array<const Def*> args(tuple_type->num_args());
            size_t i = 0;
            for (auto type : tuple_type->args())
                args[i++] = bottom(type, loc);
            agg = tuple(args, loc, agg->name);
        } else if (auto struct_app_type = agg->type()->isa<StructAppType>()) {
            Array<const Def*> args(struct_app_type->num_elems());
            size_t i = 0;
            for (auto type : struct_app_type->elems())
                args[i++] = bottom(type, loc);
            agg = struct_agg(struct_app_type, args, loc, agg->name);
        }

    }

    // TODO double-check
    if (auto aggregate = agg->isa<Aggregate>()) {
        if (auto literal = index->isa<PrimLit>()) {
            if (!agg->isa<IndefiniteArray>()) {
                Array<const Def*> args(agg->size());
                std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
                args[primlit_value<u64>(literal)] = value;
                return aggregate->rebuild(args);
            }
        }
    }

    return cse(new Insert(agg, index, value, loc, name));
}

const Def* World::select(const Def* cond, const Def* a, const Def* b, const Location& loc, const std::string& name) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type(), loc);

    if (auto lit = cond->isa<PrimLit>())
        return lit->value().get_bool() ? a : b;

    if (cond->is_not()) {
        cond = cond->as<ArithOp>()->rhs();
        std::swap(a, b);
    }

    if (a == b)
        return a;

    return cse(new Select(cond, a, b, loc, name));
}

/*
 * memory stuff
 */

const Def* World::load(const Def* mem, const Def* ptr, const Location& loc, const std::string& name) {
    if (auto store = mem->isa<Store>())
        if (store->ptr() == ptr) {
            return tuple({mem, store->val()}, loc);
    }

    if (auto global = ptr->isa<Global>()) {
        if (!global->is_mutable())
            return global->init();
    }

    if (auto ld = Load::is_out_mem(mem)) {
        if (ptr == ld->ptr())
            return ld;
    }

    return cse(new Load(mem, ptr, loc, name));
}

const Def* World::store(const Def* mem, const Def* ptr, const Def* value, const Location& loc, const std::string& name) {
    if (value->isa<Bottom>())
        return mem;

    if (auto st = mem->isa<Store>()) {
        if (ptr == st->ptr() && value == st->val())
            return st;
    }

    if (auto insert = value->isa<Insert>()) {
        if (use_lea(ptr->type()->as<PtrType>()->referenced_type())) {
            auto peeled_store = store(mem, ptr, insert->agg(), loc);
            return store(peeled_store, lea(ptr, insert->index(), insert->loc(), insert->name), insert->value(), loc, name);
        }
    }

    return cse(new Store(mem, ptr, value, loc, name));
}

const Def* World::enter(const Def* mem, const Location& loc, const std::string& name) {
    if (auto e = Enter::is_out_mem(mem))
        return e;
    return cse(new Enter(mem, loc, name));
}

const Def* World::slot(const Type* type, const Def* frame, size_t index, const Location& loc, const std::string& name) {
    return cse(new Slot(type, frame, index, loc, name));
}

const Def* World::alloc(const Type* type, const Def* mem, const Def* extra, const Location& loc, const std::string& name) {
    return cse(new Alloc(type, mem, extra, loc, name));
}

const Def* World::global(const Def* init, const Location& loc, bool is_mutable, const std::string& name) {
    return cse(new Global(init, is_mutable, loc, name));
}

const Def* World::global_immutable_string(const Location& loc, const std::string& str, const std::string& name) {
    size_t size = str.size() + 1;

    Array<const Def*> str_array(size);
    for (size_t i = 0; i != size-1; ++i)
        str_array[i] = literal_qu8(str[i], loc);
    str_array.back() = literal_qu8('\0', loc);

    return global(definite_array(str_array, loc), loc, false, name);
}

/*
 * guided partial evaluation
 */

const Def* World::run(const Def* def, const Location& loc, const std::string& name) {
    if (auto run = def->isa<Run>()) return run;
    if (auto hlt = def->isa<Hlt>()) return hlt;
    return cse(new Run(def, loc, name));
}

const Def* World::hlt(const Def* def, const Location& loc, const std::string& name) {
    if (auto hlt = def->isa<Hlt>()) return hlt;
    if (auto run = def->isa<Run>()) def = run->def();
    return cse(new Hlt(def, loc, name));
}

/*
 * continuations
 */

Continuation* World::continuation(const FnType* fn, const Location& loc, CC cc, Intrinsic intrinsic, const std::string& name) {
    auto l = new Continuation(fn, loc, cc, intrinsic, true, name);
    THORIN_CHECK_BREAK(l->gid());
    continuations_.insert(l);

    size_t i = 0;
    for (auto arg : fn->args()) {
        auto p = param(arg, l, i++);
        l->params_.push_back(p);
    }

    return l;
}

Continuation* World::basicblock(const Location& loc, const std::string& name) {
    auto bb = new Continuation(fn_type(), loc, CC::C, Intrinsic::None, false, name);
    THORIN_CHECK_BREAK(bb->gid());
    continuations_.insert(bb);
    return bb;
}

const Param* World::param(const Type* type, Continuation* continuation, size_t index, const std::string& name) {
    auto param = new Param(type, continuation, index, continuation->loc(), name);
    THORIN_CHECK_BREAK(param->gid());
    return param;
}

/*
 * misc
 */

Array<Continuation*> World::copy_continuations() const {
    Array<Continuation*> result(continuations().size());
    std::copy(continuations().begin(), continuations().end(), result.begin());
    return result;
}

const Def* World::cse_base(const PrimOp* primop) {
    THORIN_CHECK_BREAK(primop->gid());
    auto i = primops_.find(primop);
    if (i != primops_.end()) {
        primop->unregister_uses();
        delete primop;
        return *i;
    }

    const auto& p = primops_.insert(primop);
    assert_unused(p.second && "hash/equal broken");
    return primop;
}

void World::destroy(Continuation* continuation) {
    assert(continuation->num_uses() == 0);
    assert(continuation->num_args() == 0);
    continuation->destroy_body();
    continuations_.erase(continuation);
    delete continuation;
}

/*
 * optimizations
 */

void World::cleanup() { cleanup_world(*this); }

void World::opt() {
    cleanup();
    partial_evaluation(*this);
    cleanup();
    lower2cff(*this);
    clone_bodies(*this);
    mem2reg(*this);
    lift_builtins(*this);
    inliner(*this);
    lift_enters(*this);
    dead_load_opt(*this);
    cleanup();
}

/*
 * stream
 */

std::ostream& World::stream(std::ostream& os) const {
    os << "module '" << name() << "'\n\n";

    for (auto primop : primops()) {
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
