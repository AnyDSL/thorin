#include "thorin/world.h"

// for colored output
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

#include "thorin/alpha_equiv.h"
#include "thorin/def.h"
#include "thorin/error.h"
#include "thorin/normalize.h"
#include "thorin/rewrite.h"
#include "thorin/util.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"

namespace thorin {

/*
 * constructor & destructor
 */

#ifndef NDEBUG
bool World::Arena::Lock::guard_ = false;
#endif

World::World(const std::string& name)
{
    data_.name_          = name.empty() ? "module" : name;
    data_.universe_      = insert<Universe >(0, *this);
    data_.kind_star_     = insert<KindStar >(0, *this);
    data_.kind_multi_    = insert<KindMulti>(0, *this);
    data_.kind_arity_    = insert<KindArity>(0, *this);
    data_.bot_star_      = insert<Bot>(0, kind_star(), nullptr);
    data_.top_star_      = insert<Top>(0, kind_star(), nullptr);
    data_.top_arity_     = insert<Top>(0, kind_arity(), nullptr);
    data_.sigma_         = insert<Sigma>(0, kind_star(), Defs{}, nullptr)->as<Sigma>();
    data_.tuple_         = insert<Tuple>(0, sigma(), Defs{}, nullptr)->as<Tuple>();
    data_.type_mem_      = insert<Mem>(0, *this);
    data_.type_nat_      = insert<Nat>(0, *this);
    data_.type_bool_     = lit_arity(2);
    data_.lit_bool_[0]   = lit_index(2, 0);
    data_.lit_bool_[1]   = lit_index(2, 1);

    auto star = kind_star();
    auto nat = type_nat();
    auto mem = type_mem();

    // fill truth tables
    for (size_t i = 0; i != Num<Bit>; ++i) {
        data_.Bit_[i] = tuple({tuple({lit_bool(i & 0x1), lit_bool(i & 0x2)}),
                                tuple({lit_bool(i & 0x4), lit_bool(i & 0x8)})});
    }

    data_.table_not = tuple({lit_false(), lit_true ()} , {  "id"});
    data_.table_not = tuple({lit_true (), lit_false()} , { "not"});

    {   // int/sint/real: Πw: Nat. *
        auto p = pi(nat, star);
        data_.type_int_  = axiom(p, Tag:: Int, 0, { "int"});
        data_.type_sint_ = axiom(p, Tag::SInt, 0, {"sint"});
        data_.type_real_ = axiom(p, Tag::Real, 0, {"real"});
    } { // ptr: Π[T: *, as: nat]. *
        data_.type_ptr_ = axiom(nullptr, pi({star, nat}, star), Tag::Ptr, 0, {"ptr"});
    }
#define CODE(T, o) data_.T ## _[size_t(T::o)] = axiom(normalize_ ## T<T::o>, type, Tag::T, flags_t(T::o), {op2str(T::o)});
    {   // Shr: Πw: nat. Π[int w, int w]. int w
        auto type = pi(star)->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({int_w, int_w}, int_w));
        THORIN_SHR(CODE)
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
#define CODE(T, o) data_.Conv_[size_t(T::o)] = axiom(normalize_Conv<T::o>, make_type(T::o), Tag::Conv, flags_t(T::o), {op2str(T::o)});
        THORIN_CONV(CODE)
#undef Code
    } { // hlt/run: ΠT: *. ΠT. T
        auto type = pi(star)->set_domain(star);
        auto T = type->param({"T"});
        type->set_codomain(pi(T, T));
        data_.PE_[size_t(PE::hlt)] = axiom(normalize_PE<PE::hlt>, type, Tag::PE, flags_t(PE::hlt), {op2str(PE::hlt)});
        data_.PE_[size_t(PE::run)] = axiom(normalize_PE<PE::run>, type, Tag::PE, flags_t(PE::run), {op2str(PE::run)});
    } { // known: ΠT: *. ΠT. bool
        auto type = pi(star)->set_domain(star);
        auto T = type->param({"T"});
        type->set_codomain(pi(T, type_bool()));
        data_.PE_[size_t(PE::known)] = axiom(normalize_PE<PE::known>, type, Tag::PE, flags_t(PE::known), {op2str(PE::known)});
    } { // bit: Πw: nat. Π[«bool; bool», int w, int w]. int w
        auto type = pi(star)->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({arr(type_bool(), type_bool()), int_w, int_w}, int_w));
        data_.op_bit_ = axiom(normalize_bit, type, Tag::Bit, 0, {"bit"});
    } { // bitcast: Π[D: *, S: *]. ΠS. D
        auto type = pi(star)->set_domain({star, star});
        auto D = type->param(0, {"D"});
        auto S = type->param(1, {"S"});
        type->set_codomain(pi(S, D));
        data_.op_bitcast_ = axiom(normalize_bitcast, type, Tag::Bitcast, 0, {"bitcast"});
    } { // lea:, Π[s: *M, Ts: «s; *», as: nat]. Π[ptr(Ts#Heir(j)», as), i: s]. ptr(Ts#i, as)
        auto domain = sigma(universe(), 3);
        domain->set(0, kind_multi());
        domain->set(1, arr(domain->param(0, {"s"}), star));
        domain->set(2, nat);
        auto pi1 = pi(star)->set_domain(domain);
        auto s  = pi1->param(0, {"s"});
        auto Ts = pi1->param(1, {"Ts"});
        auto as = pi1->param(2, {"as"});
        auto src_ptr = type_ptr(extract(Ts, succ(s, false)), as);
        auto pi2 = pi(star)->set_domain({src_ptr, s});
        pi2->set_codomain(type_ptr(extract(Ts, pi2->param(1, {"i"})), as));
        pi1->set_codomain(pi2);
        data_.op_lea_ = axiom(normalize_lea, pi1, Tag::LEA, 0, {"lea"});
    } { // sizeof: ΠT: *. nat
        data_.op_sizeof_ = axiom(normalize_sizeof, pi(star, nat), Tag::Sizeof, 0, {"sizeof"});
    } { // load:  Π[T: *, as: nat]. Π[M, ptr(T, as)]. [M, T]
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi({mem, ptr}, sigma({mem, T})));
        data_.op_load_ = axiom(normalize_load, type, Tag::Load, 0, {"load"});
    } { // store: Π[T: *, as: nat]. Π[M, ptr(T, as), T]. M
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi({mem, ptr, T}, mem));
        data_.op_store_ = axiom(normalize_store, type, Tag::Store, 0, {"store"});
    } { // alloc: Π[T: *, as: nat]. ΠM. [M, ptr(T, as)]
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi(mem, sigma({mem, ptr})));
        data_.op_alloc_ = axiom(nullptr, type, Tag::Alloc, 0, {"alloc"});
    } { // slot: Π[T: *, as: nat]. ΠM. [M, ptr(T, as)]
        auto type = pi(star)->set_domain({star, nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi(mem, sigma({mem, ptr})));
        data_.op_slot_ = axiom(nullptr, type, Tag::Slot, 0, {"slot"});
    }
}

// must be here to avoid inclusion of some includes in world.h
World::~World() {}

/*
 * core calculus
 */

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

    if (axiom && currying_depth == 1) {
        if (auto normalize = axiom->normalizer())
            return normalize(type, callee, arg, debug(dbg));
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
    if (std::all_of(ops.begin()+1, ops.end(), [&](auto op) { return ops[0] == op; })) return arr(n, ops[0]);
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
    if (err()) {
    // TODO type-check type vs inferred type
    }

    if (auto n = ops.size()) {
        if (type->isa_structural()) {
            if (n == 1) return ops[0];
            if (std::all_of(ops.begin()+1, ops.end(), [&](auto op) { return ops[0] == op; })) return pack(n, ops[0]);
        }

        // eta rule for tuples:
        // (extract(tup, 0), extract(tup, 1), extract(tup, 2)) -> tup
        if (auto extract = ops[0]->isa<Extract>()) {
            auto tup = extract->tuple();
            bool eta = tup->type() == type;
            for (size_t i = 0; i != n && eta; ++i) {
                if (auto extract = ops[i]->isa<Extract>()) {
                    if (auto index = isa_lit<u64>(extract->index())) {
                        if (eta &= u64(i) == *index) {
                            eta &= extract->tuple() == tup;
                            continue;
                        }
                    }
                }
                eta = false;
            }

            if (eta) return tup;
        }
    }

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

const Def* World::variant(const Def* value, Debug dbg) {
    if (auto insert = value->isa<Insert>())
        return insert->index();
    return unify<Variant>(1, value->type()->arity(), value, debug(dbg));
}

const Def* World::match(const Def* arg, Defs ptrns, Debug dbg) {
#if THORIN_ENABLE_CHECKS
    assertf(ptrns.size() > 0, "match must take at least one pattern");
#endif
    const Def* type = ptrns[0]->type()->as<thorin::Case>()->codomain();
#if THORIN_ENABLE_CHECKS
    for (auto ptrn_ : ptrns) {
        assertf(ptrn_->type()->isa<thorin::Case>(), "match patterns must have 'Case' type");
        assertf(ptrn_->type()->as<thorin::Case>()->codomain() == type,
            "match cases codomains are not consistent with each other, got {} and {}",
            ptrn_->type()->as<thorin::Case>()->codomain(), type);
    }
#endif
    Array<const Def*> ops(ptrns.size() + 1);
    ops[0] = arg;
    std::copy(ptrns.begin(), ptrns.end(), ops.begin() + 1);
    // We need to build a match to have something to give to the error handler
    auto match = unify<Match>(ptrns.size() + 1, type, ops, debug(dbg));

    bool trivial = ptrns[0]->as<Ptrn>()->is_trivial();
    if (trivial) {
        if (ptrns.size() > 1 && err()) {
            for (auto ptrn : ptrns.skip_front()) {
                if (!ptrn->as<Ptrn>()->can_be_redundant())
                    err()->redundant_match_case(match, ptrn->as<Ptrn>());
            }
        }
        return ptrns[0]->as<Ptrn>()->apply(arg);
    }
    if (ptrns.size() == 1 && !trivial) {
        if (err()) err()->incomplete_match(match);
        return bot(type);
    }
    // Constant folding
    if (arg->is_const()) {
        for (auto ptrn : ptrns) {
            // If the pattern matches the argument
            if (ptrn->as<Ptrn>()->matches(arg))
                return ptrn->as<Ptrn>()->apply(arg);
        }
        return bot(type);
    }
    return match;
}

template<tag_t tag>
static const Def* merge_cmps(const Def* tuple, const Def* a, const Def* b, Debug dbg) {
    static_assert(sizeof(flags_t) == 4, "if this ever changes, please adjust the logic below");
    static constexpr size_t num_bits = log2(Num<Tag2Enum<tag>>);
    auto a_cmp = isa<tag>(a);
    auto b_cmp = isa<tag>(b);

    if (a_cmp && b_cmp && a_cmp->args() == b_cmp->args()) {
        // push flags of a_cmp and b_cmp through truth table
        flags_t res = 0;
        flags_t a_flags = a_cmp.axiom()->flags();
        flags_t b_flags = b_cmp.axiom()->flags();
        for (size_t i = 0; i != num_bits; ++i, res >>= 1, a_flags >>= 1, b_flags >>= 1)
            res |= as_lit<u32>(proj(proj(tuple, a_flags & 1), b_flags & 1)) << 31_u32;
        res >>= (31_u32 - u32(num_bits));

        auto& world = tuple->world();
        if constexpr (tag == Tag::RCmp)
            return world.op(RCmp(res), /*rmode*/ a_cmp->decurry()->arg(0), a_cmp->arg(0), a_cmp->arg(1), dbg);
        else
            return world.op(ICmp(res), a_cmp->arg(0), a_cmp->arg(1), dbg);
    }
    return nullptr;
}

const Def* World::extract(const Def* tup, const Def* index, Debug dbg) {
    if (index->isa<Arr>() || index->isa<Pack>()) {
        Array<const Def*> ops(index->lit_arity(), [&](size_t) { return extract(tup, index->ops().back()); });
        return index->isa<Arr>() ? sigma(ops, dbg) : tuple(ops, dbg);
    } else if (index->isa<Sigma>() || index->isa<Tuple>()) {
        Array<const Def*> idx(index->num_ops(), [&](size_t i) { return index->op(i); });
        Array<const Def*> ops(index->num_ops(), [&](size_t i) { return proj(tup, as_lit<nat_t>(idx[i])); });
        return index->isa<Sigma>() ? sigma(ops, dbg) : tuple(ops, dbg);
    }

    auto type = tup->unfold_type();
    assertf(alpha_equiv(type->arity(), index->type()),
            "extracting from tuple '{}' of arity '{}' with index '{}' of type '{}'", tup, type->arity(), index, index->type());

    if (isa_lit_arity(index->type(), 1)) return tup;
    if (auto pack = tup->isa<Pack>()) return pack->body();

    // extract(insert(x, index, val), index) -> val
    if (auto insert = tup->isa<Insert>()) {
        if (index == insert->index())
            return insert->val();
    }

    if (auto i = isa_lit<u64>(index)) {
        if (auto tuple = tup->isa<Tuple>()) return tuple->op(*i);

        // extract(insert(x, j, val), i) -> extract(x, i) where i != j (guaranteed by rule above)
        if (auto insert = tup->isa<Insert>()) {
            if (insert->index()->isa<Lit>())
                return extract(insert->tuple(), index, dbg);
        }

        if (type->isa<Sigma>() || type->isa<Union>())
            return unify<Extract>(2, tup->type()->op(*i), tup, index, debug(dbg));
    }

    if (auto arr = type->isa<Arr>()) {
        if (auto tuple = tup->isa<Tuple>()) {
            // TODO we could even deal with an offset
            // extract((0, 1, 2, ...), i) -> i
            bool ascending = true;
            for (size_t i = 0, e = tuple->num_ops(); i != e && ascending; ++i) {
                if (auto lit = isa_lit<u64>(tuple->op(i)))
                    ascending &= *lit == i;
                else
                    ascending = false;
            }

            if (ascending)
                return op_bitcast(arr->codomain(), index, dbg);

            // extract((a, b, c, ...), extract((..., 2, 1, 0), i)) -> extract(..., c, b, a), i
            // this also deals with NOT
            if (auto i_ex = index->isa<Extract>()) {
                if (auto i_tup = i_ex->tuple()->isa<Tuple>()) {
                    bool descending = true;
                    for (size_t i = 0, e = i_tup->num_ops(); i != e && descending; ++i) {
                        if (auto lit = isa_lit<u64>(i_tup->op(i)))
                            descending &= *lit == e - i - 1;
                        else
                            descending = false;
                    }

                    if (descending) {
                        auto ops = tuple->split();
                        std::reverse(ops.begin(), ops.end());
                        return extract(this->tuple(type, ops, tuple->debug()), i_ex->index(), dbg);
                    }
                }
            }
        }
    }

    if (auto inner = tup->isa<Extract>()) {
        auto a = inner->index();
        auto b = index;
        auto inner_type = inner->tuple()->unfold_type();
        auto arity = inner_type->lit_arity();

        if (inner->tuple()->is_const()) {
            if (auto res = merge_cmps<Tag::ICmp>(inner->tuple(), a, b, dbg)) return res;
            if (auto res = merge_cmps<Tag::RCmp>(inner->tuple(), a, b, dbg)) return res;
        }

        if (is_symmetric(inner->tuple())) {
            if (a == b) {
                // extract(extract(sym, a), a) -> extract(diag(sym), a)
                auto ops = Array<const Def*>(arity, [&](size_t i) { return proj(proj(inner->tuple(), i), i); });
                return extract(tuple(ops), a, dbg);
            } else if (a->gid() > b->gid()) {
                // extract(extract(sym, b), a) -> extract(extract(sym, a), b)
                return extract(extract(inner->tuple(), b, inner->debug()), a, dbg);
            }
        }
    }

#if 0
    if (tup == table_not()) {
        if (auto icmp = isa<Tag::ICmp>(index)) { auto [x, y] = icmp->args<2>(); return op(ICmp(~flags_t(icmp.flags()) & 0b11111), y, x, dbg); }
        if (auto rcmp = isa<Tag::RCmp>(index)) { auto [x, y] = rcmp->args<2>(); return op(RCmp(~flags_t(rcmp.flags()) & 0b01111), /*rmode*/ rcmp->decurry()->arg(0), y, x, dbg); }
    }
#endif

    // TODO absorption

    type = type->as<Arr>()->codomain();
    return unify<Extract>(2, type, tup, index, debug(dbg));
}

const Def* World::insert(const Def* tup, const Def* index, const Def* val, Debug dbg) {
    auto type = tup->unfold_type();
    assertf(alpha_equiv(type->arity(), index->type()),
            "inserting into tuple {} of arity {} with index {} of type {}", tup, type->arity(), index, index->type());

    if (isa_lit_arity(index->type(), 1)) return tuple(tup, {val}, dbg); // tup could be nominal - that's why the tuple ctor is needed

    // insert((a, b, c, d), 2, x) -> (a, b, x, d)
    if (auto t = tup->isa<Tuple>()) {
        Array<const Def*> new_ops = t->ops();
        new_ops[as_lit<u64>(index)] = val;
        return tuple(type, new_ops, dbg);
    }

    // insert(‹4; x›, 2, y) -> (x, x, y, x)
    if (auto pack = tup->isa<Pack>()) {
        if (auto a = isa_lit<u64>(pack->arity())) {
            Array<const Def*> new_ops(*a, pack->body());
            new_ops[as_lit<u64>(index)] = val;
            return tuple(type, new_ops, dbg);
        }
    }

    // insert(insert(x, index, y), index, val) -> insert(x, index, val)
    if (auto insert = tup->isa<Insert>()) {
        if (insert->index() == index)
            tup = insert->tuple();
    }

    // insert(x : U, index, y) -> insert(bot : U, index, y)
    if (tup->type()->isa<Union>())
        tup = bot(tup->type());
    return unify<Insert>(3, tup, index, val, debug(dbg));
}

const Def* World::arr(const Def* domain, const Def* codomain, Debug dbg) {
    assert(domain->type()->isa<KindArity>() || domain->type()->isa<KindMulti>());

    if (auto a = isa_lit<u64>(domain)) {
        if (*a == 0) return sigma();
        if (*a == 1) return codomain;
    }

    auto type = kind_star();
    return unify<Arr>(2, type, domain, codomain, debug(dbg));
}

const Def* World::pack(const Def* domain, const Def* body, Debug dbg) {
    assert(domain->type()->isa<KindArity>() || domain->type()->isa<KindMulti>());

    if (auto a = isa_lit<u64>(domain)) {
        if (*a == 0) return tuple();
        if (*a == 1) return body;
    }

    auto type = arr(domain, body->type());
    return unify<Pack>(1, type, body, debug(dbg));
}

const Def* World::arr(Defs domains, const Def* codomain, Debug dbg) {
    if (domains.empty()) return codomain;
    return arr(domains.skip_back(), arr(domains.back(), codomain, dbg), dbg);
}

const Def* World::pack(Defs domains, const Def* body, Debug dbg) {
    if (domains.empty()) return body;
    return pack(domains.skip_back(), pack(domains.back(), body, dbg), dbg);
}

const Def* World::succ(const Def* type, bool tuplefy, Debug dbg) {
    if (auto a = isa_lit_arity(type)) {
        Array<const Def*> ops(*a, [&](size_t i) { return lit_index(*a, i); });
        return tuplefy ? tuple(ops, dbg) : sigma(ops, dbg);
    }

    return unify<Succ>(0, type, tuplefy, debug(dbg));
}

const Lit* World::lit_index(const Def* a, u64 i, Debug dbg) {
    if (a->isa<Top>()) return lit(a, i, dbg);

    auto arity = as_lit<u64>(a);
    if (err() && i >= arity) err()->index_out_of_range(arity, i);

    return lit(a, i, dbg);
}

const Def* World::bot_top(bool is_top, const Def* type, Debug dbg) {
    if (auto arr = type->isa<Arr>()) return pack(arr->domain(), bot_top(is_top, arr->codomain()), dbg);
    if (auto sigma = type->isa<Sigma>())
        return tuple(sigma, Array<const Def*>(sigma->num_ops(), [&](size_t i) { return bot_top(is_top, sigma->op(i), dbg); }), dbg);
    auto d = debug(dbg);
    return is_top ? (const Def*) unify<Top>(0, type, d) : (const Def*) unify<Bot>(0, type, d);
}

const Def* World::cps2ds(const Def* cps, Debug dbg) {
    if (auto ds = cps->isa<DS2CPS>())
        return ds->ds();
    auto cn  = cps->type()->as<Pi>();
    auto ret = cn->domain()->as<Sigma>()->op(cn->num_domains() - 1)->as<Pi>();
    auto type = pi(sigma(cn->domains().skip_back()), ret->domain());
    return unify<CPS2DS>(1, type, cps, debug(dbg));
}

const Def* World::ds2cps(const Def* ds, Debug dbg) {
    if (auto cps = ds->isa<CPS2DS>())
        return cps->cps();
    auto fn  = ds->type()->as<Pi>();
    Array<const Def*> domains(fn->num_domains() + 1);
    for (size_t i = 0, n = fn->num_domains(); i < n; ++i)
        domains[i] = fn->domain(i);
    domains.back() = cn(fn->codomain());
    auto type = cn(domains);
    return unify<DS2CPS>(1, type, ds, debug(dbg));
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
    if (auto sigma = t->isa<Sigma>()) return world.tuple(sigma->ops());
    if (auto arr   = t->isa<Arr  >()) return world.pack(arr->domain(), arr->codomain());
    return t;
}

const Def* World::op_lea(const Def* ptr, const Def* index, Debug dbg) {
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();
    auto Ts = tuple_of_types(pointee);
    return app(app(op_lea(), {pointee->arity(), Ts, addr_space}), {ptr, index}, debug(dbg));
}

const Def* World::op_cast(const Def* dst_type, const Def* src, Debug dbg) {
    if (isa_lit_arity(src->type(), 2) && get_width(dst_type))
        src = op_bitcast(type_int(1), src); // cast bool first to i1

    if (auto _int = isa<Tag::Int>(src->type())) {
        if (false) {}
        else if (auto _int = isa<Tag:: Int>(dst_type)) return     op(Conv::u2u, dst_type, src, dbg);
        else if (auto sint = isa<Tag::SInt>(dst_type)) return tos(op(Conv::u2u, dst_type, src, dbg));
        else if (auto real = isa<Tag::Real>(dst_type)) return     op(Conv::u2r, dst_type, src, dbg);
    } else if (auto sint = isa<Tag::SInt>(src->type())) {
        src = toi(src);
        if (false) {}
        else if (auto _int = isa<Tag:: Int>(dst_type)) return     op(Conv::s2s, dst_type, src, dbg);
        else if (auto sint = isa<Tag::SInt>(dst_type)) return tos(op(Conv::s2s, dst_type, src, dbg));
        else if (auto real = isa<Tag::Real>(dst_type)) return     op(Conv::s2r, dst_type, src, dbg);
    } else if (auto real = isa<Tag::Real>(src->type())) {
        if (false) {}
        else if (auto _int = isa<Tag:: Int>(dst_type)) return     op(Conv::r2u, dst_type, src, dbg);
        else if (auto sint = isa<Tag::SInt>(dst_type)) return tos(op(Conv::r2s, dst_type, src, dbg));
        else if (auto real = isa<Tag::Real>(dst_type)) return     op(Conv::r2r, dst_type, src, dbg);
    }

    return op_bitcast(dst_type, src, dbg);
}

const Def* World::op(Cmp cmp, const Def* a, const Def* b, Debug dbg) {
    if (isa_lit_arity(a->type(), 2)) {
        switch (cmp) {
            case Cmp::eq: return extract_eq(a, b, dbg);
            case Cmp::ne: return extract_ne(a, b, dbg);
            default: THORIN_UNREACHABLE;
        }
    } else if (auto _int = isa<Tag::Int>(a->type())) {
        switch (cmp) {
            case Cmp::eq: return op(ICmp::  e, a, b, dbg);
            case Cmp::ne: return op(ICmp:: ne, a, b, dbg);
            case Cmp::lt: return op(ICmp::ul , a, b, dbg);
            case Cmp::le: return op(ICmp::ule, a, b, dbg);
            case Cmp::gt: return op(ICmp::ug , a, b, dbg);
            case Cmp::ge: return op(ICmp::uge, a, b, dbg);
            default: THORIN_UNREACHABLE;
        }
    } else if (auto sint = isa<Tag::SInt>(a->type())) {
        switch (cmp) {
            case Cmp::eq: return op(ICmp::  e, toi(a), toi(b), dbg);
            case Cmp::ne: return op(ICmp:: ne, toi(a), toi(b), dbg);
            case Cmp::lt: return op(ICmp::sl , toi(a), toi(b), dbg);
            case Cmp::le: return op(ICmp::sle, toi(a), toi(b), dbg);
            case Cmp::gt: return op(ICmp::sg , toi(a), toi(b), dbg);
            case Cmp::ge: return op(ICmp::sge, toi(a), toi(b), dbg);
            default: THORIN_UNREACHABLE;
        }
    } else if (auto real = isa<Tag::Real>(a->type())) {
        // TODO for now, use RMode::none
        switch (cmp) {
            case Cmp::eq: return op(RCmp::  e, RMode::none, a, b, dbg);
            case Cmp::ne: return op(RCmp::une, RMode::none, a, b, dbg);
            case Cmp::lt: return op(RCmp:: l , RMode::none, a, b, dbg);
            case Cmp::le: return op(RCmp:: le, RMode::none, a, b, dbg);
            case Cmp::gt: return op(RCmp:: g , RMode::none, a, b, dbg);
            case Cmp::ge: return op(RCmp:: ge, RMode::none, a, b, dbg);
            default: THORIN_UNREACHABLE;
        }
    } else if (isa<Tag::Ptr>(a->type())) {
        auto x = op_bitcast(type_int(64), a);
        auto y = op_bitcast(type_int(64), b);
        switch (cmp) {
            case Cmp::eq: return op(ICmp:: e, x, y, dbg);
            case Cmp::ne: return op(ICmp::ne, x, y, dbg);
            default: THORIN_UNREACHABLE;
        }
    }
    THORIN_UNREACHABLE;
}

/*
 * misc
 */

std::vector<Lam*> World::copy_lams() const {
    std::vector<Lam*> result;

    for (auto def : data_.defs_) {
        if (auto lam = def->isa_nominal<Lam>())
            result.emplace_back(lam);
    }

    return result;
}

#if THORIN_ENABLE_CHECKS

const Def* World::lookup_by_gid(u32 gid) {
    auto i = std::find_if(data_.defs_.begin(), data_.defs_.end(), [&](const Def* def) { return def->gid() == gid; });
    if (i == data_.defs_.end()) return nullptr;
    return *i;
}

#endif

/*
 * visit & rewrite
 */

template<bool elide_empty>
void World::visit(VisitFn f) const {
    unique_queue<NomSet> noms;

    for (const auto& [name, nom] : externals()) {
        assert(nom->is_set() && "external must not be empty");
        noms.push(nom);
    }

    while (!noms.empty()) {
        auto nom = noms.pop();
        if (elide_empty && !nom->is_set()) continue;
        Scope scope(nom);
        f(scope);
        scope.visit({}, {}, {}, {}, [&](const Def* def) {
            if (nom = def->isa_nominal(); nom && !scope.contains(nom)) noms.push(nom);
        });
    }
}

void World::rewrite(const std::string& info, EnterFn enter_fn, RewriteFn rewrite_fn) {
    ILOG("{}: start,", info);

    visit([&](const Scope& scope) {
        if (enter_fn(scope)) {
            auto& s = const_cast<Scope&>(scope); // yes, we know what we are doing
            if (s.rewrite(info, rewrite_fn)) s.update();
        } else {
            VLOG("{}: skipping scope {}", info, scope.entry());
        }
    });

    ILOG("{}: done,", info);
}

/*
 * misc
 */

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
std::string Log::colorize(const std::string& str, int) {
#endif
    return str;
}

void World::set(std::unique_ptr<ErrorHandler>&& err) { err_ = std::move(err); }

Stream& World::stream(Stream& s) const {
    s << "module '" << name() << "'\n\n";

    std::vector<const Global*> globals;

    for (auto def : defs()) {
        if (auto global = def->isa<Global>())
            globals.emplace_back(global);
    }

    for (auto global : globals)
        stream_assignment(s, global).endl();

    visit<false>([&] (const Scope& scope) {
        if (scope.entry()->isa<Axiom>()) return;
        scope.stream(s);
    });
    return s;
}

template void Streamable<World>::write(const std::string& filename) const;
template void Streamable<World>::write() const;
template void Streamable<World>::dump() const;
template void World::visit<true> (VisitFn) const;
template void World::visit<false>(VisitFn) const;

}
