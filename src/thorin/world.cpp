#include "thorin/world.h"

#include <fstream>

#include "thorin/alpha_equiv.h"
#include "thorin/def.h"
#include "thorin/normalize.h"
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

World::World(uint32_t cur_gid, const std::string& name, bool tuple2pack)
    : root_page_(new Zone)
    , cur_page_(root_page_.get())
    , name_(name.empty() ? "module" : name)
    , cur_gid_(cur_gid)
    , tuple2pack_(tuple2pack)
{
    cache_.universe_      = insert<Universe >(0, *this);
    cache_.kind_star_     = insert<KindStar >(0, *this);
    cache_.kind_multi_    = insert<KindMulti>(0, *this);
    cache_.kind_arity_    = insert<KindArity>(0, *this);
    cache_.bot_star_      = insert<Bot>(0, kind_star(), nullptr);
    cache_.top_star_      = insert<Top>(0, kind_star(), nullptr);
    cache_.top_arity_     = insert<Top>(0, kind_arity(), nullptr);
    cache_.sigma_         = insert<Sigma>(0, kind_star(), Defs{}, nullptr)->as<Sigma>();
    cache_.tuple_         = insert<Tuple>(0, sigma(), Defs{}, nullptr)->as<Tuple>();
    cache_.type_mem_      = insert<Mem>(0, *this);
    cache_.type_nat_      = insert<Nat>(0, *this);
    cache_.lit_arity_1_   = lit_arity(1);
    cache_.lit_index_0_1_ = lit_index(lit_arity_1(), 0);
    cache_.op_end_        = axiom(bot_star(), Tag::End, 0, {"end"});

    auto star = kind_star();
    auto nat = type_nat();
    auto mem = type_mem();

    {   // int/real: Πw: Nat. *
        auto p = pi(nat, star);
        cache_.type_int_  = axiom(p, Tag::Int,  0, {"int"});
        cache_.type_real_ = axiom(p, Tag::Real, 0, {"real"});
        cache_.type_bool_ = type_int(1);
        cache_.lit_bool_[0] = lit(type_bool(), false);
        cache_.lit_bool_[1] = lit(type_bool(),  true);
    } { // ptr: Π[T: *, as: nat]. *
        cache_.type_ptr_ = axiom(pi({star, nat}, star), Tag::Ptr, 0, {"ptr"});
    }
#define CODE(T, o) cache_.T ## _[size_t(T::o)] = axiom(normalize_ ## T<T::o>, type, 0, Tag::T, flags_t(T::o), {op2str(T::o)});
    {   // IOp: Πw: nat. Π[int w, int w]. int w
        auto type = pi(star)->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({int_w, int_w}, int_w));
        THORIN_I_OP(CODE)
    } { // WOp: Π[m: nat, w: nat]. Π[int w, int w]. int w
        auto type = pi(star)->set_domain({nat, nat});
        type->param(0, {"m"});
        auto int_w = type_int(type->param(1, {"w"}));
        type->set_codomain(pi({int_w, int_w}, int_w));
        THORIN_W_OP(CODE)
    } { // ZOp: Πw: nat. Π[mem, int w, int w]. [mem, int w]
        auto type = pi(star)->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({mem, int_w, int_w}, sigma({mem, int_w})));
        THORIN_Z_OP(CODE)
    } { // ROp: Π[m: nat, w: nat]. Π[real w, real w]. real w
        auto type = pi(star)->set_domain({nat, nat});
        type->param(0, {"m"});
        auto real_w = type_real(type->param(1, {"w"}));
        type->set_codomain(pi({real_w, real_w}, real_w));
        THORIN_R_OP(CODE)
    } { // ICmp: Πw: nat. Π[int w, int w]. bool
        auto type = pi(star)->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({int_w, int_w}, type_bool()));
        THORIN_I_CMP(CODE)
    } { // RCmp: Π[m: nat, w: nat]. Π[real w, real w]. bool
        auto type = pi(star)->set_domain({nat, nat});
        type->param(0, {"m"});
        auto real_w = type_real(type->param(1, {"w"}));
        type->set_codomain(pi({real_w, real_w}, type_bool()));
        THORIN_R_CMP(CODE)
    }
#undef CODE
    {   // Conv: Π[dw: nat, sw: nat]. Πi/r sw. i/r dw
        auto make_type = [&](Conv o) {
            auto type = pi(star)->set_domain({nat, nat});
            auto dw = type->param(0, {"dw"});
            auto sw = type->param(1, {"sw"});
            auto type_dw = o == Conv::s2r || o == Conv::u2r || o == Conv::r2r ? type_real(dw) : type_int(dw);
            auto type_sw = o == Conv::r2s || o == Conv::r2u || o == Conv::r2r ? type_real(sw) : type_int(sw);
            return type->set_codomain(pi(type_sw, type_dw));
        };
#define CODE(T, o) cache_.Conv_[size_t(T::o)] = axiom(normalize_Conv<T::o>, make_type(T::o), 0, Tag::Conv, flags_t(T::o), {op2str(T::o)});
        THORIN_CONV(CODE)
#undef Code
    } { // hlt/run: ΠT: *. ΠT. T
        auto type = pi(star)->set_domain(star);
        auto T = type->param({"T"});
        type->set_codomain(pi(T, T));
        cache_.PE_[size_t(PE::hlt)] = axiom(normalize_PE<PE::hlt>, type, 0, Tag::PE, flags_t(PE::hlt), {op2str(PE::hlt)});
        cache_.PE_[size_t(PE::run)] = axiom(normalize_PE<PE::run>, type, 0, Tag::PE, flags_t(PE::run), {op2str(PE::run)});
    } { // known: ΠT: *. ΠT. bool
        auto type = pi(star)->set_domain(star);
        auto T = type->param({"T"});
        type->set_codomain(pi(T, type_bool()));
        cache_.PE_[size_t(PE::known)] = axiom(normalize_PE<PE::known>, type, 0, Tag::PE, flags_t(PE::known), {op2str(PE::known)});
    } { // bitcast: Π[D: *, S: *]. ΠS. D
        auto type = pi(star)->set_domain({star, star});
        auto D = type->param(0, {"D"});
        auto S = type->param(1, {"S"});
        type->set_codomain(pi(S, D));
        cache_.op_bitcast_ = axiom(normalize_bitcast, type, 0, Tag::Bitcast, 0, {"bitcast"});
    } { // select: ΠT: *. Π[bool, T, T]. T
        auto type = pi(star)->set_domain(star);
        auto T = type->param({"T"});
        cache_.op_select_ = axiom(normalize_select, type->set_codomain(pi({type_bool(), T, T}, T)), 0, Tag::Select, 0, {"select"});
    } { // lea:, Π[s: *M, Ts: «s; *», as: nat]. Π[ptr(«j: s; Ts#j», as), i: s]. ptr(Ts#i, as)
        auto domain = sigma(universe(), 3);
        domain->set(0, kind_multi());
        domain->set(1, variadic(domain->param(0, {"s"}), star));
        domain->set(2, nat);
        auto pi1 = pi(star)->set_domain(domain);
        auto s  = pi1->param(0, {"s"});
        auto Ts = pi1->param(1, {"Ts"});
        auto as = pi1->param(2, {"as"});
        auto v = variadic(star)->set_domain(s);
        v->set_codomain(extract(Ts, v->param({"j"})));
        auto src_ptr = type_ptr(v, as);
        auto pi2 = pi(star)->set_domain({src_ptr, s});
        pi2->set_codomain(type_ptr(extract(Ts, pi2->param(1, {"i"})), as));
        pi1->set_codomain(pi2);
        cache_.op_lea_ = axiom(normalize_lea, pi1, 0 , Tag::LEA, 0, {"lea"});
    } { // sizeof: ΠT: *. nat
        cache_.op_sizeof_ = axiom(normalize_sizeof, pi(star, nat), 0, Tag::Sizeof, 0, {"sizeof"});
    } { // load:  Π[T: *, as: nat]. Π[M, ptr(T, as)]. [M, T]
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi({mem, ptr}, sigma({mem, T})));
        cache_.op_load_ = axiom(normalize_load, type, 0, Tag::Load, 0, {"load"});
    } { // store: Π[T: *, as: nat]. Π[M, ptr(T, as), T]. M
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi({mem, ptr, T}, mem));
        cache_.op_store_ = axiom(normalize_store, type, 0, Tag::Store, 0, {"store"});
    } { // alloc: Π[T: *, as: nat]. ΠM. [M, ptr(T, as)]
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi(mem, sigma({mem, ptr})));
        cache_.op_alloc_ = axiom(nullptr, type, 0, Tag::Alloc, 0, {"alloc"});
    } { // slot: Π[T: *, as: nat]. ΠM. [M, ptr(T, as)]
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi(mem, sigma({mem, ptr})));
        cache_.op_slot_ = axiom(nullptr, type, 0, Tag::Slot, 0, {"slot"});
    }
}

/*
 * core calculus
 */

Axiom* World::axiom(Def::NormalizeFn normalize, const Def* type, size_t num_ops, tag_t tag, flags_t flags, Debug dbg) {
    auto a = insert<Axiom>(num_ops, normalize, type, num_ops, tag, flags, debug(dbg));
    a->make_external();
    assert(lookup(a->name()) == a);
    return a;
}

static const Def* lub(const Def* t1, const Def* t2) { // TODO broken
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

const Lam* World::lam(const Def* domain, const Def* filter, const Def* body, Debug dbg) {
    auto p = pi(domain, body->type());
    return unify<Lam>(2, p, filter, body, debug(dbg));
}

const Def* World::app(const Def* callee, const Def* arg, Debug dbg) {
    auto pi = callee->type()->as<Pi>();
    auto type = pi->apply(arg);

    auto [axiom, currying_depth] = get_axiom(callee); // TODO move down again
#if 0
    if (axiom == nullptr || (axiom->tag() != Tag::Bitcast && axiom->tag() != Tag::LEA)) // HACK
        assertf(pi->domain() == arg->type(), "callee '{}' expects an argument of type '{}' but the argument '{}' is of type '{}'\n", callee, pi->domain(), arg, arg->type());
#endif

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

    if (axiom && currying_depth == 1) {
        if (auto normalize = axiom->normalizer()) {
            return normalize(type, callee, arg, debug(dbg));
        }
    }

    return unify<App>(2, axiom, currying_depth-1, type, callee, arg, debug(dbg));
}

const Def* World::raw_app(const Def* callee, const Def* arg, Debug dbg) {
    auto pi = callee->type()->as<Pi>();
    auto type = pi->apply(arg);
    auto [axiom, currying_depth] = get_axiom(callee);
    return unify<App>(2, axiom, currying_depth-1, type, callee, arg, debug(dbg));
}

const Def* World::sigma(const Def* type, Defs ops, Debug dbg) {
    auto n = ops.size();
    if (n == 0) return sigma();
    if (n == 1) return ops[0];
    if (tuple2pack_ && std::all_of(ops.begin()+1, ops.end(), [&](auto op) { return ops[0] == op; }))
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

    if (tuple2pack_ && std::all_of(ops.begin()+1, ops.end(), [&](auto op) { return ops[0] == op; }))
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

const Def* World::union_(const Def* type, Defs ops, Debug dbg) {
    assertf(ops.size() > 0, "unions must have at least one operand");
    if (ops.size() == 1) return ops[0];
    // Remove duplicate operands
    Array<const Def*> ops_copy(ops);
    std::sort(ops_copy.begin(), ops_copy.end());
    ops.skip_back(ops_copy.end() - std::unique(ops_copy.begin(), ops_copy.end()));
    return unify<Union>(ops_copy.size(), type, ops_copy, debug(dbg));
}

const Def* World::variant_(const Def* type, const Def* index, const Def* arg, Debug dbg) {
#if THORIN_ENABLE_CHECKS
    // TODO:
    // - assert that 'type', when reduced, is a 'union' with 'type->arity() == index->type()'
    // - assert that 'type', when reduced, is a 'union' with 'type->op(index) == arg->type()'
#endif
    return unify<Variant_>(2, type, index, arg, debug(dbg));
}

const Def* World::variant_(const Def* type, const Def* arg, Debug dbg) {
    // TODO: reduce 'type'
    assertf(type->isa<Union>() && !type->isa_nominal(), "only nominal unions can be created with this constructor");
    size_t index = std::find(type->ops().begin(), type->ops().end(), arg->type()) - type->ops().begin();
    assertf(index != type->num_ops(), "cannot find type {} in union {}", arg->type(), type);
    return variant_(type, lit_index(index, type->num_ops()), arg, dbg);
}

const Def* World::match_(const Def* arg, Defs cases, Debug dbg) {
#if THORIN_ENABLE_CHECKS
    assertf(cases.size() > 0, "match must take at least one case");
    assertf(cases[0]->type()->isa<Pi>(), "match cases must be functions");
#endif
    auto type = cases[0]->type()->as<Pi>()->codomain();
#if THORIN_ENABLE_CHECKS
    for (auto case_ : cases) {
        assertf(case_->type()->isa<Pi>(), "match cases must be functions");
        assertf(case_->type()->as<Pi>()->codomain() == type,
            "match cases codomains are not consistent with each other, got {} and {}",
            case_->type()->as<Pi>()->codomain(), type);
    }
    // TODO:
    // - assert that `arg->type()`, when reduced, is a `union` with arity == cases.size()
#endif
    if (auto variant = arg->isa<Variant_>())
        return app(cases[as_lit<nat_t>(variant->index())], variant->arg());
    Array<const Def*> ops(cases.size() + 1);
    ops[0] = arg;
    std::copy(cases.begin(), cases.end(), ops.begin() + 1);
    return unify<Match_>(cases.size() + 1, type, ops, debug(dbg));
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

    auto type = agg->type()->as<Variadic>()->codomain();
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

const Def* World::variadic(const Def* domain, const Def* codomain, Debug dbg) {
    assert(domain->type()->isa<KindArity>() || domain->type()->isa<KindMulti>());

    if (auto a = isa_lit<u64>(domain)) {
        if (*a == 0) return sigma();
        if (*a == 1) return codomain;
    }

    auto type = kind_star();
    return unify<Variadic>(2, type, domain, codomain, debug(dbg));
}

const Def* World::pack(const Def* domain, const Def* body, Debug dbg) {
    assert(domain->type()->isa<KindArity>() || domain->type()->isa<KindMulti>());

    if (auto a = isa_lit<u64>(domain)) {
        if (*a == 0) return tuple();
        if (*a == 1) return body;
    }

    auto type = variadic(domain, body->type());
    return unify<Pack>(1, type, body, debug(dbg));
}

const Def* World::variadic(Defs domains, const Def* codomain, Debug dbg) {
    if (domains.empty()) return codomain;
    return variadic(domains.skip_back(), variadic(domains.back(), codomain, dbg), dbg);
}

const Def* World::pack(Defs domains, const Def* body, Debug dbg) {
    if (domains.empty()) return body;
    return pack(domains.skip_back(), pack(domains.back(), body, dbg), dbg);
}

const Lit* World::lit_index(const Def* a, u64 i, Debug dbg) {
    if (a->isa<Top>()) return lit(a, i, dbg);

    auto arity = as_lit<u64>(a);
    assertf(i < arity, "index literal '{}' does not fit within arity '{}'", i, a);

    return lit(a, i, dbg);
}

const Def* World::bot_top(bool is_top, const Def* type, Debug dbg) {
    if (auto variadic = type->isa<Variadic>()) return pack(variadic->domain(), bot_top(is_top, variadic->codomain()), dbg);
    if (auto sigma = type->isa<Sigma>())
        return tuple(sigma, Array<const Def*>(sigma->num_ops(), [&](size_t i) { return bot_top(is_top, sigma->op(i), dbg); }), dbg);
    auto d = debug(dbg);
    return is_top ? (const Def*) unify<Top>(0, type, d) : (const Def*) unify<Bot>(0, type, d);
}

const Def* World::cps2ds(const Def* cps, Debug dbg) {
    auto cn  = cps->type()->as<Pi>();
    auto ret = cn->domain()->as<Sigma>()->op(cn->num_domains() - 1)->as<Pi>();
    auto type = pi(sigma(cn->domains().skip_back()), ret->domain());
    return unify<CPS2DS>(1, type, cps, debug(dbg));
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
 * ops
 */

static const Def* tuple_of_types(const Def* t) {
    auto& world = t->world();
    if (auto sigma    = t->isa<Sigma>()) return world.tuple(sigma->ops());
    if (auto variadic = t->isa<Variadic>()) return world.pack(variadic->domain(), variadic->codomain());
    return t; // Variadic might be nominal. Thus, we still might have this case.
}

const Def* World::op_lea(const Def* ptr, const Def* index, Debug dbg) {
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();
    auto Ts = tuple_of_types(pointee);
    //Ts->dump();
    return app(app(op_lea(), {pointee->arity(), Ts, addr_space}), {ptr, index}, debug(dbg));
}

/*
 * deprecated
 */

#if 0
/*
 * arithops
 */

const Def* World::arithop(ArithOpTag tag, const Def* a, const Def* b, Debug dbg) {
    assert(a->type() == b->type());
    auto type = a->type();

    auto llit = a->isa<Lit>();
    auto rlit = b->isa<Lit>();

    if (is_type_i(type) || type == PrimType_bool) {
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
        }

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

    return unify<ArithOp>(2, tag, a, b, debug(dbg));
}
#endif

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

#if THORIN_ENABLE_CHECKS

const Def* World::lookup_by_gid(u32 gid) {
    auto i = std::find_if(defs_.begin(), defs_.end(), [&](const Def* def) { return def->gid() == gid; });
    if (i == defs_.end()) return nullptr;
    return *i;
}

#endif

/*
 * visit & rewrite
 */

template<bool elide_empty>
void World::visit(VisitFn f) const {
    unique_queue<NomSet> nom_queue;

    for (const auto& [name, nom] : externals()) {
        assert(nom->is_set() && "external must not be empty");
        nom_queue.push(nom);
    }

    while (!nom_queue.empty()) {
        auto nom = nom_queue.pop();
        if (elide_empty && !nom->is_set()) continue;
        Scope scope(nom);
        f(scope);

        unique_queue<DefSet> def_queue;
        for (auto def : scope.free())
            def_queue.push(def);

        while (!def_queue.empty()) {
            auto def = def_queue.pop();
            if (auto nom = def->isa_nominal())
                nom_queue.push(nom);
            else {
                for (auto op : def->ops())
                    def_queue.push(op);
            }
        }
    }
}

void World::rewrite(const std::string& info, EnterFn enter_fn, RewriteFn rewrite_fn) {
    VLOG("start: {},", info);

    visit([&](const Scope& scope) {
        if (enter_fn(scope)) {
            auto new_body = thorin::rewrite(scope.entry(), scope, rewrite_fn);

            if (scope.entry()->ops().back() != new_body) {
                scope.entry()->set(scope.entry()->num_ops()-1, new_body);
                const_cast<Scope&>(scope).update(); // yes, we know what we are doing
            }
        }
    });
    VLOG("end: {},", info);
}

template void World::visit<true> (VisitFn) const;
template void World::visit<false>(VisitFn) const;

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

    //for (auto global : globals)
        //global->stream_assignment(os);

    visit<false>([&] (const Scope& scope) {
        if (scope.entry()->isa<Axiom>()) return;
        scope.stream(os);
    });
    return os;
}

void World::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }

void World::thorin() const {
    auto filename = std::string(name()) + ".thorin";
    write_thorin(filename.c_str());
}

}
