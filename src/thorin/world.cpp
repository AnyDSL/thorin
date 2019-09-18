#include "thorin/world.h"

#include <fstream>

#include "thorin/alpha_equiv.h"
#include "thorin/def.h"
#include "thorin/normalize.h"
#include "thorin/primop.h"
#include "thorin/rewrite.h"
#include "thorin/util.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"
#include "thorin/util/log.h"

namespace thorin {

const Def* infer_width(const Def* def) {
    auto app = def->type()->as<App>();
    assert(isa<Tag::Int>(def->type()) || isa<Tag::Real>(def->type()));
    return app->arg();
}

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
    cache_.universe_      = insert<Universe>(0, *this);
    cache_.kind_arity_    = insert<KindArity>(0, *this);
    cache_.kind_multi_    = insert<KindMulti>(0, *this);
    cache_.kind_star_     = insert<KindStar >(0, *this);
    cache_.bot_star_      = insert<Bot>(0, kind_star(), nullptr);
    cache_.top_arity_     = insert<Top>(0, kind_arity(), nullptr);
    cache_.sigma_         = insert<Sigma>(0, kind_star(), Defs{}, nullptr)->as<Sigma>();
    cache_.tuple_         = insert<Tuple>(0, sigma(), Defs{}, nullptr)->as<Tuple>();
    cache_.type_mem_      = insert<Mem>(0, *this);
    cache_.type_nat_      = insert<Nat>(0, *this);
    cache_.lit_arity_1_   = lit_arity(1);
    cache_.lit_index_0_1_ = lit_index(lit_arity_1(), 0);
    cache_.end_scope_     = lam(cn(), Lam::CC::C, Lam::Intrinsic::EndScope, {"end_scope"});

    {   // int/real: Πw: Nat. *
        auto p = pi(type_nat(), kind_star());
        cache_.type_int_  = axiom(p, Tag::Int,  0, {"int"});
        cache_.type_real_ = axiom(p, Tag::Real, 0, {"real"});
        cache_.lit_bool_[0] = lit(type_bool(), false);
        cache_.lit_bool_[1] = lit(type_bool(),  true);
    }

    // TODO
    // lea:,  Π[s: *M, Ts: «s; *», as: nat]. Π[ptr(«j: s; Ts#j», as), i: s]. ptr(Ts#i, as)
    // load:  Π[T: *, as: nat]. Π[M as, ptr(T, as)]. [M as, T]
    // store: Π[T: *, as: nat]. Π[M as, ptr(T, as), T]. M as
    // enter: Πas: nat. ΠM as. [M as, F as]
    // slot:  Π[T: *, as: nat]. Π[F as, nat]. ptr(T, as)

    { // analyze: Π[s: *M, Ts: «s; *», T: *]. Π[nat, «i: s; Ts#i»], T
        auto domain = sigma(universe(), 3);
        domain->set(0, kind_multi());
        domain->set(1, variadic(domain->param(0, {"s"}), kind_star()));
        domain->set(2, kind_star());
        auto type = pi(kind_star())->set_domain(domain);
        auto v = variadic(kind_star())->set_arity(type->param(0, {"s"}));
        auto i = v->param({"i"});
        v->set_body(extract(type->param(1, {"Ts"}), i));
        type->set_codomain(pi({type_nat(), v}, type->param(2, {"T"})));
        type->dump();
        cache_.op_analyze_ = axiom(type, Tag::Analyze, 0, {"analyze"});
    } { // bitcast: Π[S: *, D: *]. ΠS. D
        auto type = pi(kind_star())->set_domain({kind_star(), kind_star()});
        auto S = type->param(0, {"S"});
        auto D = type->param(1, {"D"});
        type->set_codomain(pi(S, D));
        cache_.op_bitcast_ = axiom(normalize_bitcast, type, 0, Tag::Bitcast, 0, {"bitcast"});
    } { // select: ΠT:*. Π[bool, T, T]. T
        auto type = pi(kind_star())->set_domain(kind_star());
        auto T = type->param({"T"});
        cache_.op_select_ = axiom(normalize_select, type->set_codomain(pi({type_bool(), T, T}, T)), 0, Tag::Select, 0, {"select"});
    } { // sizeof: ΠT:*. nat
        cache_.op_sizeof_ = axiom(normalize_sizeof, pi(kind_star(), type_nat()), 0, Tag::Sizeof, 0, {"sizeof"});
    }
#define CODE(T, o) cache_.T ## _[size_t(T::o)] = axiom(normalize_ ## T<T::o>, type, 0, Tag::T, flags_t(T::o), {op2str(T::o)});
    {   // IOp: Πw: nat. Π[int w, int w]. int w
        auto type = pi(kind_star())->set_domain(type_nat());
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({int_w, int_w}, int_w));
        THORIN_I_OP(CODE)
    } { // WOp: Π[m: nat, w: nat]. Π[int w, int w]. int w
        auto type = pi(kind_star())->set_domain({type_nat(), type_nat()});
        type->param(0, {"m"});
        auto int_w = type_int(type->param(1, {"w"}));
        type->set_codomain(pi({int_w, int_w}, int_w));
        THORIN_W_OP(CODE)
    } { // ZOp: Πw: nat. Π[mem, int w, int w]. [mem, int w]
        auto type = pi(kind_star())->set_domain(type_nat());
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({type_mem(), int_w, int_w}, sigma({type_mem(), int_w})));
        THORIN_Z_OP(CODE)
    } { // ROp: Π[m: nat, w: nat]. Π[real w, real w]. real w
        auto type = pi(kind_star())->set_domain({type_nat(), type_nat()});
        type->param(0, {"m"});
        auto real_w = type_real(type->param(1, {"w"}));
        type->set_codomain(pi({real_w, real_w}, real_w));
        THORIN_R_OP(CODE)
    } { // ICmp: Πw: nat. Π[int w, int w]. bool
        auto type = pi(kind_star())->set_domain(type_nat());
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({int_w, int_w}, type_bool()));
        THORIN_I_CMP(CODE)
    } { // RCmp: Π[m: nat, w: nat]. Π[real w, real w]. bool
        auto type = pi(kind_star())->set_domain({type_nat(), type_nat()});
        type->param(0, {"m"});
        auto real_w = type_real(type->param(1, {"w"}));
        type->set_codomain(pi({real_w, real_w}, type_bool()));
        THORIN_R_CMP(CODE)
    }
#undef CODE
#define CODE(T, o) \
    {   /* Conv: Π[sw: nat, dw: nat]. Πi/r sw. i/r dw */                                                          \
        auto type = pi(kind_star())->set_domain({type_nat(), type_nat()});                                        \
        auto sw = type->param(0, {"sw"});                                                                         \
        auto dw = type->param(1, {"dw"});                                                                         \
        auto type_sw = T::o == T::r2s || T::o == T::r2u || T::o == T::r2r ? type_real(sw) : type_int(sw);         \
        auto type_dw = T::o == T::s2r || T::o == T::u2r || T::o == T::r2r ? type_real(dw) : type_int(dw);         \
        type->set_codomain(pi(type_sw, type_dw));                                                                 \
        cache_.Conv_[size_t(T::o)] = axiom(normalize_Conv<T::o>, type, 0, Tag::Conv, flags_t(T::o), {op2str(T::o)}); \
    }
    THORIN_CONV(CODE)
#undef Code
}

World::~World() {
    for (auto def : defs_) def->~Def();
}

Axiom* World::axiom(Def::NormalizeFn normalize, const Def* type, size_t num_ops, tag_t tag, flags_t flags, Debug dbg) {
    auto a = insert<Axiom>(0, normalize, type, num_ops, tag, flags, debug(dbg));
    a->make_external();
    assert(lookup(a->name()) == a);
    return a;
}

bool assignable(const Def* dst, const Def* src) {
    if (dst == src) return true;
    return false;
}

const Def* World::app(const Def* callee, const Def* arg, Debug dbg) {
    auto pi = callee->type()->as<Pi>();
    auto type = pi->apply(arg);

    //assertf(pi->domain() == arg->type(), "callee '{}' expects an argument of type '{}' but the argument '{}' is of type '{}'\n", callee, type, arg, arg->type());

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

    auto [axiom, currying_depth] = get_axiom(callee);

    if (axiom && currying_depth == 1) {
        if (auto normalize = axiom->normalizer()) {
            if (auto normalized = normalize(type, callee, arg, debug(dbg)))
                return normalized;
        }
    }

    return unify<App>(2, axiom, currying_depth-1, type, callee, arg, debug(dbg));
}

const Lam* World::lam(const Def* domain, const Def* filter, const Def* body, Debug dbg) {
    auto p = pi(domain, body->type());
    return unify<Lam>(2, p, filter, body, debug(dbg));
}

static const Def* lub(const Def* t1, const Def* t2) {
    if (t1->isa<Universe>()) return t1;
    if (t2->isa<Universe>()) return t2;
    //assert(t1->isa<Kind>() && t2->isa<Kind>());
    switch (std::max(t1->node(), t2->node())) {
        case Node::KindArity: return t1->world().kind_arity();
        case Node::KindMulti: return t1->world().kind_multi();
        case Node::KindStar:  return t1->world().kind_star();
        default: THORIN_UNREACHABLE;
    }
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
        ops.emplace_back(lit_nat(*s));
    return tuple(ops, dbg);
}

const Def* World::extract(const Def* agg, const Def* index, Debug dbg) {
    assertf(alpha_equiv(agg->type()->arity(), index->type()),
            "extracting from aggregate {} of arity {} with index {} of type {}", agg, agg->type()->arity(), index, index->type());

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
    assertf(alpha_equiv(agg->type()->arity(), index->type()),
            "inserting into aggregate {} of arity {} with index {} of type {}", agg, agg->type()->arity(), index, index->type());

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

const Lit* World::lit_index(const Def* a, u64 i, Debug dbg) {
    if (a->isa<Top>()) return lit(a, i, dbg);

    auto arity = as_lit<u64>(a);
    assertf(i < arity, "index literal '{}' does not fit within arity '{}'", i, a);

    return lit(a, i, dbg);
}

#if 0
/*
 * arithops
 */

const Def* World::arithop(ArithOpTag tag, const Def* a, const Def* b, Debug dbg) {
    assert(a->type() == b->type());
    auto type = a->type();

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
#endif

const Def* World::bot_top(bool is_top, const Def* type, Debug dbg) {
    if (auto variadic = type->isa<Variadic>()) return pack(variadic->arity(), bot_top(is_top, variadic->body()), dbg);
    if (auto sigma = type->isa<Sigma>())
        return tuple(sigma, Array<const Def*>(sigma->num_ops(), [&](size_t i) { return bot_top(is_top, sigma->op(i), dbg); }), dbg);
    auto d = debug(dbg);
    return is_top ? (const Def*) unify<Top>(0, type, d) : (const Def*) unify<Bot>(0, type, d);
}

/*
 * aggregate operations
 */

const Def* World::lea(const Def* ptr, const Def* index, Debug dbg) {
    auto type_ptr = ptr->type()->as<Ptr>();
    auto pointee = type_ptr->pointee();

    assertf(pointee->arity() == index->type(), "lea of aggregate {} of arity {} with index {} of type {}", pointee, pointee->arity(), index, index->type());

    if (pointee->arity() == lit_arity_1()) return ptr;

    const Def* type = nullptr;
    if (auto sigma = pointee->isa<Sigma>()) {
        type = this->type_ptr(sigma->op(as_lit<u64>(index)), type_ptr->addr_space());
    } else {
        auto variadic = pointee->as<Variadic>();
        type = this->type_ptr(variadic->body(), type_ptr->addr_space());
    }

    return unify<LEA>(2, type, ptr, index, debug(dbg));
}

/*
 * memory stuff
 */

const Def* World::load(const Def* mem, const Def* ptr, Debug dbg) {
    auto pointee = ptr->type()->as<Ptr>()->pointee();

    // loading an empty tuple can only result in an empty tuple
    if (auto sigma = pointee->isa<Sigma>(); sigma && sigma->num_ops() == 0)
        return tuple({mem, tuple(sigma->type(), {}, dbg)});

    return unify<Load>(2, sigma({type_mem(), pointee}), mem, ptr, debug(dbg));
}

const Def* World::store(const Def* mem, const Def* ptr, const Def* val, Debug dbg) {
    if (val->isa<Bot>()) return mem;
    if (auto pack = val->isa<Pack>(); pack && pack->body()->isa<Bot>()) return mem;
    if (auto tuple = val->isa<Tuple>()) {
        if (std::all_of(tuple->ops().begin(), tuple->ops().end(), [&](const Def* op) { return op->isa<Bot>(); }))
            return mem;
    }

    assert(ptr->type()->as<Ptr>()->pointee() == val->type());
    return unify<Store>(3, mem, ptr, val, debug(dbg));
}

const Alloc* World::alloc(const Def* type, const Def* mem, Debug dbg) {
    return unify<Alloc>(1, sigma({type_mem(), type_ptr(type)}), mem, debug(dbg));
}

const Slot* World::slot(const Def* type, const Def* mem, Debug dbg) {
    return unify<Slot>(1, sigma({type_mem(), type_ptr(type)}), mem, debug(dbg));
}

const Def* World::global(const Def* id, const Def* init, bool is_mutable, Debug dbg) {
    return unify<Global>(2, type_ptr(init->type()), id, init, is_mutable, debug(dbg));
}

const Def* World::global_immutable_string(const std::string& str, Debug dbg) {
    size_t size = str.size() + 1;

    Array<const Def*> str_array(size);
    for (size_t i = 0; i != size-1; ++i)
        str_array[i] = lit_nat(str[i], dbg);
    str_array.back() = lit_nat('\0', dbg);

    return global(tuple(str_array, dbg), false, dbg);
}

/*
const Assembly* World::assembly(const Def* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg) {
    return unify<Assembly>(inputs.size(), type, inputs, asm_template, output_constraints, input_constraints, clobbers, flags, debug(dbg))->as<Assembly>();;
}

const Assembly* World::assembly(Defs types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg) {
    Array<const Def*> output(types.size()+1);
    std::copy(types.begin(), types.end(), output.begin()+1);
    output.front() = type_mem();

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
    if (pe_done_ || def->isa<Hlt>()) return lit_bool(false);
    if (is_const(def)) return lit_bool(true);

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

    std::vector<const Global*> globals;

    for (auto def : defs()) {
        if (auto global = def->isa<Global>())
            globals.emplace_back(global);
    }

    for (auto global : globals)
        global->stream_assignment(os);

    Scope::for_each<false>(*this, [&] (const Scope& scope) { scope.stream(os); });
    return os;
}

void World::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }

void World::thorin() const {
    auto filename = std::string(name()) + ".thorin";
    write_thorin(filename.c_str());
}

}
