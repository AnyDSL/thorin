#include "thorin/world.h"

#include <fstream>

#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/rewrite.h"
#include "thorin/util.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"
#include "thorin/util/log.h"

namespace thorin {

/*
 * constructor and destructor
 */

#ifndef NDEBUG
bool World::Lock::allocate_guard_ = false;
#endif

World::World(uint32_t cur_gid, const std::string& name)
    : root_page_(new Zone)
    , cur_page_(root_page_.get())
    , name_(name.empty() ? "module" : name)
    , cur_gid_(cur_gid)
{
    cache_.universe_         = insert<Universe>(0, *this);
    cache_.kind_.kind_arity_ = insert<Kind>(0, *this, Node_KindArity);
    cache_.kind_.kind_multi_ = insert<Kind>(0, *this, Node_KindMulti);
    cache_.kind_.kind_star_  = insert<Kind>(0, *this, Node_KindStar);
#define THORIN_ALL_TYPE(T, M) \
    cache_.primtype_.T##_    = insert<PrimType>(0, *this, PrimType_##T);
#include "thorin/tables/primtypetable.h"
    cache_.bot_star_         = insert<BotTop>(0, false, kind_star(), nullptr);
    cache_.top_arity_        = insert<BotTop>(0, true, kind_arity(), nullptr);
    cache_.sigma_            = insert<Sigma>(0, kind_star(), Defs{}, nullptr)->as<Sigma>();
    cache_.tuple_            = insert<Tuple>(0, sigma(), Defs{}, nullptr)->as<Tuple>();
    cache_.mem_              = insert<MemType>(0, *this);
    cache_.lit_arity_1_      = lit_arity(1);
    cache_.lit_index_0_1_    = lit_index(lit_arity_1(), 0);
    cache_.lit_bool_[0]      = lit(type_bool(), {false});
    cache_.lit_bool_[1]      = lit(type_bool(), {true});
    cache_.end_scope_        = lam(cn(), Lam::CC::C, Lam::Intrinsic::EndScope, {"end_scope"});
}

World::~World() {
    for (auto def : defs_) def->~Def();
}

const Def* World::app(const Def* callee, const Def* arg, Debug dbg) {
    auto pi = callee->type()->as<Pi>();
    assertf(pi->domain() == arg->type(), "callee '{}' expects an argument of type '{}' but the argument '{}' is of type '{}'\n", callee, pi->domain(), arg, arg->type());

    if (auto lam = callee->isa<Lam>()) {
        if (lam->intrinsic() == Lam::Intrinsic::Match) {
            auto args = arg->as<Tuple>()->ops();
            if (args.size() == 2) return app(args[1], Defs{}, dbg);
            if (auto lit = args[0]->isa<Lit>()) {
                for (size_t i = 2; i < args.size(); i++) {
                    if (extract(args[i], 0_s)->as<Lit>() == lit)
                        return app(extract(args[i], 1), Defs{}, dbg);
                }
                return app(args[1], Defs{}, dbg);
            }
        }
    }

    return unify<App>(2, pi->codomain(), callee, arg, debug(dbg));
}

const Lam* World::lam(const Def* domain, const Def* filter, const Def* body, Debug dbg) {
    auto p = pi(domain, body->type());
    return unify<Lam>(2, p, filter, body, debug(dbg));
}

static const Def* lub(const Def* t1, const Def* t2) {
    if (t1->isa<Universe>()) return t1;
    if (t2->isa<Universe>()) return t2;
    assert(t1->isa<Kind>() && t2->isa<Kind>());
    auto tag = std::max(t1->tag(), t2->tag());
    return t1->world().kind(tag);
}

const Pi* World::pi(const Def* domain, const Def* codomain, Debug dbg) {
    auto type = lub(domain->type(), codomain->type());
    return unify<Pi>(2, type, domain, codomain, debug(dbg));
}

const Def* World::sigma(const Def* type, Defs ops, Debug dbg) {
    auto n = ops.size();
    if (n == 0) return sigma();
    if (n == 1) return ops[0];
    if (std::all_of(ops.begin()+1, ops.end(), [&](auto op) { return ops[0] == op; }))
        return variadic(n, ops[0]);
    return unify<Sigma>(ops.size(), type, ops, debug(dbg));
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

    auto n = ops.size();
    if (n == 0) return tuple();
    if (n == 1) return ops[0];
    if (type->isa_nominal()) return unify<Tuple>(ops.size(), type, ops, debug(dbg));

    if (std::all_of(ops.begin()+1, ops.end(), [&](auto op) { return ops[0] == op; }))
        return pack(n, ops[0]);

    // eta rule for tuples:
    // (extract(agg, 0), extract(agg, 1), extract(agg, 2)) -> agg
    bool eta = true;
    const Def* agg = nullptr;
    for (size_t i = 0; i != n && eta; ++i) {
        if (auto extract = ops[i]->isa<Extract>()) {
            if (auto index = isa_lit<u64>(extract->index())) {
                if (eta &= u64(i) == *index) {
                    if (i == 0) {
                        agg = extract->agg();
                        eta &= agg->type() == type;
                    } else {
                        eta &= extract->agg() == agg;
                    }
                    continue;
                }
            }
        }
        eta = false;
    }

    if (eta) return agg;
    return unify<Tuple>(ops.size(), type, ops, debug(dbg));
}

const Def* World::tuple_str(const char* s, Debug dbg) {
    std::vector<const Def*> ops;
    for (; *s != '\0'; ++s)
        ops.emplace_back(lit_qs8(*s));
    return tuple(ops, dbg);
}

const Def* World::extract(const Def* agg, const Def* index, Debug dbg) {
    assertf(agg->arity() == index->type(), "extracting from aggregate {} of arity {} with index {} of type {}", agg, agg->arity(), index, index->type());

    if (index->type() == lit_arity_1()) return agg;
    if (auto pack = agg->isa<Pack>()) return pack->body();

    // extract(insert(x, index, val), index) -> val
    if (auto insert = agg->isa<Insert>()) {
        if (index == insert->index())
            return insert->val();
    }

    if (auto i = isa_lit<u64>(index)) {
        if (auto tuple = agg->isa<Tuple>()) return tuple->op(*i);

        // extract(insert(x, j, val), i) -> extract(x, i) where i != j (guaranteed by rule above)
        if (auto insert = agg->isa<Insert>()) {
            if (insert->index()->isa<Lit>())
                return extract(insert->agg(), index, dbg);
        }

        if (auto sigma = agg->type()->isa<Sigma>())
            return unify<Extract>(2, sigma->op(*i), agg, index, debug(dbg));
    }

    auto type = agg->type()->as<Variadic>()->body();
    return unify<Extract>(2, type, agg, index, debug(dbg));
}

const Def* World::insert(const Def* agg, const Def* index, const Def* val, Debug dbg) {
    assertf(agg->arity() == index->type(), "inserting into aggregate {} of arity {} with index {} of type {}", agg, agg->arity(), index, index->type());

    if (index->type() == lit_arity_1()) return val;

    // insert((a, b, c, d), 2, x) -> (a, b, x, d)
    if (auto tup = agg->isa<Tuple>()) {
        Array<const Def*> new_ops = tup->ops();
        new_ops[as_lit<u64>(index)] = val;
        return tuple(tup->type(), new_ops, dbg);
    }

    // insert(‹4; x›, 2, y) -> (x, x, y, x)
    if (auto pack = agg->isa<Pack>()) {
        if (auto a = isa_lit<u64>(pack->arity())) {
            Array<const Def*> new_ops(*a, pack->body());
            new_ops[as_lit<u64>(index)] = val;
            return tuple(pack->type(), new_ops, dbg);
        }
    }

    // insert(insert(x, index, y), index, val) -> insert(x, index, val)
    if (auto insert = agg->isa<Insert>()) {
        if (insert->index() == index)
            agg = insert->agg();
    }

    return unify<Insert>(3, agg, index, val, debug(dbg));
}

const Def* World::variadic(const Def* arity, const Def* body, Debug dbg) {
    if (auto a = isa_lit<u64>(arity)) {
        if (*a == 0) return sigma();
        if (*a == 1) return body;
    }

    auto type = kind_star();
    return unify<Variadic>(2, type, arity, body, debug(dbg));
}

const Def* World::pack(const Def* arity, const Def* body, Debug dbg) {
    if (auto a = isa_lit<u64>(arity)) {
        if (*a == 0) return tuple();
        if (*a == 1) return body;
    }

    auto type = variadic(arity, body->type());
    return unify<Pack>(1, type, body, debug(dbg));
}

const Def* World::variadic(Defs arity, const Def* body, Debug dbg) {
    if (arity.empty()) return body;
    return variadic(arity.skip_back(), variadic(arity.back(), body, dbg), dbg);
}

const Def* World::pack(Defs arity, const Def* body, Debug dbg) {
    if (arity.empty()) return body;
    return pack(arity.skip_back(), pack(arity.back(), body, dbg), dbg);
}

/*
 * literals
 */

const Lit* World::lit_allset(PrimTypeTag tag, Debug dbg) {
    switch (tag) {
#define THORIN_I_TYPE(T, M) \
    case PrimType_##T: return lit(PrimType_##T, std::numeric_limits<u64>::max(), dbg);
#define THORIN_BOOL_TYPE(T, M) \
    case PrimType_##T: return lit(PrimType_##T, u64(true), dbg);
#include "thorin/tables/primtypetable.h"
        default: THORIN_UNREACHABLE;
    }
}

const Lit* World::lit_index(const Def* a, u64 i, Debug dbg) {
    if (is_top(a)) return lit(a, i, dbg);

    auto arity = as_lit<u64>(a);
    assertf(i < arity, "index literal '{}' does not fit within arity '{}'", i, a);

    return lit(a, i, dbg);
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
    PrimTypeTag type = a->type()->as<PrimType>()->primtype_tag();

    auto llit = a->isa<Lit>();
    auto rlit = b->isa<Lit>();

    if (llit && rlit) {

        try {
            switch (tag) {
                case ArithOp_add:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() + rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_sub:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() - rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_mul:
                    switch (type) {
#define THORIN_P_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() * rlit->get<T>()), dbg);
#define THORIN_Q_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() * rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_div:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() / rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_rem:
                    switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() % rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_and:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return lit(type, T(llit->get<T>() & rlit->get<T>()), dbg);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() & rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_or:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return lit(type, T(llit->get<T>() | rlit->get<T>()), dbg);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() | rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_xor:
                    switch (type) {
#define THORIN_I_TYPE(T, M)    case PrimType_##T: return lit(type, T(llit->get<T>() ^ rlit->get<T>()), dbg);
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() ^ rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shl:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() << rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
                case ArithOp_shr:
                    switch (type) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: return lit(type, T(llit->get<T>() >> rlit->get<T>()), dbg);
#include "thorin/tables/primtypetable.h"
                        default: THORIN_UNREACHABLE;
                    }
            }
        } catch (BottomException) {
            return bot(type, dbg);
        }
    }

    // normalize: swap literal to the left
    if (is_commutative(tag) && rlit) {
        std::swap(a, b);
        std::swap(llit, rlit);
    }

    if (is_type_i(type) || type == PrimType_bool) {
        if (a == b) {
            switch (tag) {
                case ArithOp_add: return arithop_mul(lit(type, 2, dbg), a, dbg);

                case ArithOp_sub:
                case ArithOp_xor: return lit_zero(type, dbg);

                case ArithOp_and:
                case ArithOp_or:  return a;

                case ArithOp_div:
                    if (is_zero(b))
                        return bot(type, dbg);
                    return lit_one(type, dbg);

                case ArithOp_rem:
                    if (is_zero(b))
                        return bot(type, dbg);
                    return lit_zero(type, dbg);

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
                case ArithOp_shr: return lit_zero(type, dbg);

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
                case ArithOp_rem: return bot(type, dbg);

                case ArithOp_shl:
                case ArithOp_shr: return a;

                default: break;
            }
        }

        if (is_one(b)) {
            switch (tag) {
                case ArithOp_mul:
                case ArithOp_div: return a;
                case ArithOp_rem: return lit_zero(type, dbg);

                default: break;
            }
        }

        if (rlit && as_lit<u64>(rlit) >= uint64_t(num_bits(type))) {
            switch (tag) {
                case ArithOp_shl:
                case ArithOp_shr: return bot(type, dbg);

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
                return lit_bool(true, dbg);

        if (tag == ArithOp_and && lcmp && rcmp && lcmp->lhs() == rcmp->lhs() && lcmp->rhs() == rcmp->rhs()
                && lcmp->cmp_tag() == negate(rcmp->cmp_tag()))
                return lit_bool(false, dbg);

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

    // normalize: try to reorder same ops to have the literal on the left-most side
    if (is_associative(tag) && is_type_i(a->type())) {
        auto a_same = a->isa<ArithOp>() && a->as<ArithOp>()->arithop_tag() == tag ? a->as<ArithOp>() : nullptr;
        auto b_same = b->isa<ArithOp>() && b->as<ArithOp>()->arithop_tag() == tag ? b->as<ArithOp>() : nullptr;
        auto a_lhs_lv = a_same && a_same->lhs()->isa<Lit>() ? a_same->lhs() : nullptr;
        auto b_lhs_lv = b_same && b_same->lhs()->isa<Lit>() ? b_same->lhs() : nullptr;

        if (is_commutative(tag)) {
            if (a_lhs_lv && b_lhs_lv)
                return arithop(tag, arithop(tag, a_lhs_lv, b_lhs_lv, dbg), arithop(tag, a_same->rhs(), b_same->rhs(), dbg), dbg);
            if (llit && b_lhs_lv)
                return arithop(tag, arithop(tag, a, b_lhs_lv, dbg), b_same->rhs(), dbg);
            if (b_lhs_lv)
                return arithop(tag, b_lhs_lv, arithop(tag, a, b_same->rhs(), dbg), dbg);
        }
        if (a_lhs_lv)
            return arithop(tag, a_lhs_lv, arithop(tag, a_same->rhs(), b, dbg), dbg);
    }

    return unify<ArithOp>(2, tag, a, b, debug(dbg));
}

const Def* World::arithop_not(const Def* def, Debug dbg) { return arithop_xor(lit_allset(def->type(), dbg), def, dbg); }

const Def* World::arithop_minus(const Def* def, Debug dbg) {
    switch (PrimTypeTag tag = def->type()->as<PrimType>()->primtype_tag()) {
#define THORIN_F_TYPE(T, M) \
        case PrimType_##T: \
            return arithop_sub(lit(PrimType_##T, M(-0.f), dbg), def, dbg);
#include "thorin/tables/primtypetable.h"
        default:
            assert(is_type_i(tag));
            return arithop_sub(lit_zero(tag, dbg), def, dbg);
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

    if (is_bot(a) || is_bot(b)) return bot(type_bool(), dbg);

    if (oldtag != tag)
        std::swap(a, b);

    auto llit = a->isa<Lit>();
    auto rlit = b->isa<Lit>();

    if (llit && rlit) {
        PrimTypeTag type = llit->type()->as<PrimType>()->primtype_tag();

        switch (tag) {
            case Cmp_eq:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_bool(llit->get<T>() == rlit->get<T>(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case Cmp_ne:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_bool(llit->get<T>() != rlit->get<T>(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case Cmp_lt:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_bool(llit->get<T>() <  rlit->get<T>(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case Cmp_le:
                switch (type) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_bool(llit->get<T>() <= rlit->get<T>(), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            default: THORIN_UNREACHABLE;
        }
    }

    if (a == b) {
        switch (tag) {
            case Cmp_lt:
            case Cmp_ne: return lit_zero(type_bool(), dbg);
            case Cmp_le:
            case Cmp_eq: return lit_one(type_bool(), dbg);
            default: break;
        }
    }

    return unify<Cmp>(2, tag, a, b, debug(dbg));
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
    if (is_bot(from)) return bot(to);
    if (from->type() == to) return from;

    if (auto variant = from->isa<Variant>()) {
        if (variant->op(0)->type() != to)
            ELOG("variant downcast not possible");
        return variant->op(0);
    }

    auto lit = from->isa<Lit>();
    auto to_type = to->isa<PrimType>();
    if (lit && to_type) {
        switch (lit->type()->as<PrimType>()->primtype_tag()) {
            case PrimType_bool:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<bool>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps8:
            case PrimType_qs8:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<s8>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps16:
            case PrimType_qs16:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<s16>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps32:
            case PrimType_qs32:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<s32>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_ps64:
            case PrimType_qs64:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<s64>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu8:
            case PrimType_qu8:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<u8>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu16:
            case PrimType_qu16:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<u16>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu32:
            case PrimType_qu32:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<u32>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pu64:
            case PrimType_qu64:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<u64>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pf16:
            case PrimType_qf16:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<f16>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pf32:
            case PrimType_qf32:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<f32>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
            case PrimType_pf64:
            case PrimType_qf64:
                switch (to_type->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return lit_##T(M(lit->get<f64>()), dbg);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
        }
    }

    if (lit && is_arity(to))
        return lit_index(to, lit->get());

    return unify<Cast>(1, to, from, debug(dbg));
}

const Def* World::bitcast(const Def* to, const Def* from, Debug dbg) {
    if (is_bot(from)) return bot(to);
    if (from->type() == to) return from;

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
        if (auto lit = from->isa<Lit>())
            return this->lit(prim_to->primtype_tag(), lit->get(), dbg);
    }

    return unify<Bitcast>(1, to, from, debug(dbg));
}

const Def* World::bot_top(bool is_top, const Def* type, Debug dbg) {
    if (auto variadic = type->isa<Variadic>()) return pack(variadic->arity(), bot_top(is_top, variadic->body()), dbg);
    if (auto sigma = type->isa<Sigma>())
        return tuple(sigma, Array<const Def*>(sigma->num_ops(), [&](size_t i) { return bot_top(is_top, sigma->op(i), dbg); }), dbg);
    return unify<BotTop>(0, is_top, type, debug(dbg));
}

/*
 * aggregate operations
 */

const Def* World::lea(const Def* ptr, const Def* index, Debug dbg) {
    auto ptr_type = ptr->type()->as<PtrType>();
    auto pointee = ptr_type->pointee();

    assertf(pointee->arity() == index->type(), "lea of aggregate {} of arity {} with index {} of type {}", pointee, pointee->arity(), index, index->type());

    if (pointee->arity() == lit_arity_1()) return ptr;

    const Def* type = nullptr;
    if (auto sigma = pointee->isa<Sigma>()) {
        type = this->ptr_type(sigma->op(as_lit<u64>(index)), ptr_type->addr_space());
    } else {
        auto variadic = pointee->as<Variadic>();
        type = this->ptr_type(variadic->body(), ptr_type->addr_space());
    }

    return unify<LEA>(2, type, ptr, index, debug(dbg));
}

const Def* World::select(const Def* cond, const Def* a, const Def* b, Debug dbg) {
    if (is_bot(cond) || is_bot(a) || is_bot(b)) return bot(a->type(), dbg);
    if (auto lit = cond->isa<Lit>()) return lit->get<bool>() ? a : b;

    if (is_not(cond)) {
        cond = cond->as<ArithOp>()->rhs();
        std::swap(a, b);
    }

    if (a == b) return a;
    return unify<Select>(3, cond, a, b, debug(dbg));
}

const Def* World::size_of(const Def* type, Debug dbg) {
    if (auto ptype = type->isa<PrimType>())
        return lit_qs32(num_bits(ptype->primtype_tag()) / 8, dbg);

    return unify<SizeOf>(1, bot(type, dbg), debug(dbg));
}

/*
 * memory stuff
 */

const Def* World::load(const Def* mem, const Def* ptr, Debug dbg) {
    auto pointee = ptr->type()->as<PtrType>()->pointee();

    // loading an empty tuple can only result in an empty tuple
    if (auto sigma = pointee->isa<Sigma>(); sigma && sigma->num_ops() == 0)
        return tuple({mem, tuple(sigma->type(), {}, dbg)});

    return unify<Load>(2, sigma({mem_type(), pointee}), mem, ptr, debug(dbg));
}

const Def* World::store(const Def* mem, const Def* ptr, const Def* val, Debug dbg) {
    if (is_bot(val)) return mem;
    if (auto pack = val->isa<Pack>(); pack && is_bot(pack->body())) return mem;
    if (auto tuple = val->isa<Tuple>()) {
        if (std::all_of(tuple->ops().begin(), tuple->ops().end(), [&](auto op) { return is_bot(op); }))
            return mem;
    }

    assert(ptr->type()->as<PtrType>()->pointee() == val->type());
    return unify<Store>(3, mem, ptr, val, debug(dbg));
}

const Alloc* World::alloc(const Def* type, const Def* mem, Debug dbg) {
    return unify<Alloc>(1, sigma({mem_type(), ptr_type(type)}), mem, debug(dbg));
}

const Slot* World::slot(const Def* type, const Def* mem, Debug dbg) {
    return unify<Slot>(1, sigma({mem_type(), ptr_type(type)}), mem, debug(dbg));
}

const Def* World::global(const Def* id, const Def* init, bool is_mutable, Debug dbg) {
    return unify<Global>(2, ptr_type(init->type()), id, init, is_mutable, debug(dbg));
}

const Def* World::global_immutable_string(const std::string& str, Debug dbg) {
    size_t size = str.size() + 1;

    Array<const Def*> str_array(size);
    for (size_t i = 0; i != size-1; ++i)
        str_array[i] = lit_qu8(str[i], dbg);
    str_array.back() = lit_qu8('\0', dbg);

    return global(tuple(str_array, dbg), false, dbg);
}

/*
const Assembly* World::assembly(const Def* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg) {
    return unify<Assembly>(inputs.size(), type, inputs, asm_template, output_constraints, input_constraints, clobbers, flags, debug(dbg))->as<Assembly>();;
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
*/

/*
 * partial evaluation related stuff
 */

const Def* World::hlt(const Def* def, Debug dbg) {
    if (pe_done_)
        return def;
    return unify<Hlt>(1, def, debug(dbg));
}

const Def* World::known(const Def* def, Debug dbg) {
    if (pe_done_ || def->isa<Hlt>())
        return lit_bool(false, dbg);
    if (is_const(def))
        return lit_bool(true, dbg);
    return unify<Known>(1, def, debug(dbg));
}

const Def* World::run(const Def* def, Debug dbg) {
    if (pe_done_)
        return def;
    return unify<Run>(1, def, debug(dbg));
}

Lam* World::match(const Def* type, size_t num_patterns) {
    Array<const Def*> arg_types(num_patterns + 2);
    arg_types[0] = type;
    arg_types[1] = cn();
    for (size_t i = 0; i < num_patterns; i++)
        arg_types[i + 2] = sigma({type, cn()});
    auto dbg = Debug("match");
    return lam(cn(sigma(arg_types)), Lam::CC::C, Lam::Intrinsic::Match, dbg);
}

/*
 * misc
 */

std::vector<Lam*> World::copy_lams() const {
    std::vector<Lam*> result;

    for (auto def : defs_) {
        if (auto lam = def->isa_nominal<Lam>())
            result.emplace_back(lam);
    }

    return result;
}

/*
 * stream
 */

std::ostream& World::stream(std::ostream& os) const {
    os << "module '" << name() << "'\n\n";

    for (auto def : defs()) {
        if (auto global = def->isa<Global>())
            global->stream_assignment(os);
    }

    Scope::for_each<false>(*this, [&] (const Scope& scope) { scope.stream(os); });
    return os;
}

void World::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }

void World::thorin() const {
    auto filename = std::string(name()) + ".thorin";
    write_thorin(filename.c_str());
}

}
