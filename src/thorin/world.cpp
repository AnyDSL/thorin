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

// TODO not all cases implemented
// TODO move somewhere else and polish

static bool check(const Def* t1, const Def* t2) {
    if (t1 == t2) return true;
    auto& world = t1->world();

    if (t1->isa<Top>() || t2->isa<Top>()) return check(t1->type(), t2->type());
    if (auto sig = t1->isa<Sigma>()) {
        if (!check(t1->arity(), t2->arity())) return false;

        auto a = t1->num_ops();
        for (size_t i = 0; i != a; ++i) {
            if (!check(sig->op(i), proj(t2, a, i))) return false;
        }

        return true;
    } else if (auto arr = t1->isa<Arr>()) {
        if (!check(t1->arity(), t2->arity())) return false;

        if (auto a = isa_lit(arr->shape())) {
            for (size_t i = 0; i != a; ++i) {
                if (!check(arr->apply(world.lit_int(*a, i)).back(), proj(t2, *a, i))) return false;
            }

            return true;
        }
    }

    if (t1->node() == t2->node() && t1->num_ops() == t2->num_ops()) {
        size_t n = t1->num_ops();
        for (size_t i = 0; i != n; ++i) {
            if (!check(t1->op(i), t2->op(i))) return false;
        }

        return true;
    }

    return false;
}

static bool assignable(const Def* type, const Def* val) {
    if (type == val->type()) return true;
    auto& world = type->world();

    if (auto sigma = type->isa<Sigma>()) {
        if (!check(type->arity(), val->type()->arity())) return false;

        auto red = sigma->apply(val);
        for (size_t i = 0, e = red.size(); i != e; ++i) {
            if (!assignable(red[i], val->out(i))) return false;
        }

        return true;
    } else if (auto arr = type->isa<Arr>()) {
        if (!check(type->arity(), val->type()->arity())) return false;

        if (auto n = isa_lit(arr->arity())) {;
            for (size_t i = 0; i != *n; ++i) {
                if (!assignable(arr->apply(world.lit_int(*n, i)).back(), val->out(i))) return false;
            }
        } else {
            return check(arr, val->type());
        }

        return true;
    } else {
        return check(type, val->type());
    }

    return false;
}

/*
 * constructor & destructor
 */

#ifndef NDEBUG
bool World::Arena::Lock::guard_ = false;
#endif

World::World(const std::string& name) {
    data_.name_          = name.empty() ? "module" : name;
    data_.universe_      = insert<Universe >(0, *this);
    data_.kind_          = insert<Kind>(0, *this);
    data_.bot_kind_      = insert<Bot>(0, kind(), nullptr);
    data_.top_kind_      = insert<Top>(0, kind(), nullptr);
    data_.sigma_         = insert<Sigma>(0, kind(), Defs{}, nullptr)->as<Sigma>();
    data_.tuple_         = insert<Tuple>(0, sigma(), Defs{}, nullptr)->as<Tuple>();
    auto nat = data_.type_nat_ = insert<Nat>(0, *this);
    data_.top_nat_       = insert<Top>(0, type_nat(), nullptr);

    {   // int/real: Πw: Nat. *
        auto p = pi(nat, kind());
        data_.type_int_     = axiom(p, Tag:: Int, 0);
        data_.type_real_    = axiom(p, Tag::Real, 0);
        data_.type_bool_    = type_int(2);
        data_.lit_bool_[0]  = lit_int(2, 0_u64);
        data_.lit_bool_[1]  = lit_int(2, 1_u64);

        data_.table_not = tuple({lit_false(), lit_true ()}, {  "id"});
        data_.table_not = tuple({lit_true (), lit_false()}, { "not"});

        // fill truth tables
        for (size_t i = 0; i != Num<Bit>; ++i) {
            data_.Bit_[i] = tuple({tuple({lit_bool(i & 0b0001), lit_bool(i & 0b0010)}),
                                   tuple({lit_bool(i & 0b0100), lit_bool(i & 0b1000)})});
        }
    }

    auto mem = data_.type_mem_ = axiom(kind(), Tag::Mem, 0, {"mem"});

    { // ptr: Π[T: *, as: nat]. *
        data_.type_ptr_ = axiom(nullptr, pi({kind(), nat}, kind()), Tag::Ptr, 0, {"ptr"});
    }
#define CODE(T, o) data_.T ## _[size_t(T::o)] = axiom(normalize_ ## T<T::o>, type, Tag::T, flags_t(T::o), {op2str(T::o)});
    {   // Shr: Πw: nat. Π[int w, int w]. int w
        auto type = pi(kind())->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({int_w, int_w}, int_w));
        THORIN_SHR(CODE)
    } { // WOp: Π[m: nat, w: nat]. Π[int w, int w]. int w
        // WOp: Π[m: nat, w: nat]. Π[r: nat, s: «r; nat»]. Π[«s; int w, «s; int w»]. «s; int w»
        auto type = pi(kind())->set_domain({nat, nat});
        type->param(0, {"m"});
        auto int_w = type_int(type->param(1, {"w"}));
        type->set_codomain(pi({int_w, int_w}, int_w));
        THORIN_W_OP(CODE)
    } { // ZOp: Πw: nat. Π[mem, int w, int w]. [mem, int w]
        auto type = pi(kind())->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({mem, int_w, int_w}, sigma({mem, int_w})));
        THORIN_Z_OP(CODE)
    } { // ROp: Π[m: nat, w: nat]. Π[real w, real w]. real w
        auto type = pi(kind())->set_domain({nat, nat});
        type->param(0, {"m"});
        auto real_w = type_real(type->param(1, {"w"}));
        type->set_codomain(pi({real_w, real_w}, real_w));
        THORIN_R_OP(CODE)
    } { // ICmp: Πw: nat. Π[int w, int w]. bool
        auto type = pi(kind())->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({int_w, int_w}, type_bool()));
        THORIN_I_CMP(CODE)
    } { // RCmp: Π[m: nat, w: nat]. Π[real w, real w]. bool
        auto type = pi(kind())->set_domain({nat, nat});
        type->param(0, {"m"});
        auto real_w = type_real(type->param(1, {"w"}));
        type->set_codomain(pi({real_w, real_w}, type_bool()));
        THORIN_R_CMP(CODE)
    }
#undef CODE
    {   // Conv: Π[dw: nat, sw: nat]. Πi/r sw. i/r dw
        auto make_type = [&](Conv o) {
            auto type = pi(kind())->set_domain({nat, nat});
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
        auto type = pi(kind())->set_domain(kind());
        auto T = type->param({"T"});
        type->set_codomain(pi(T, T));
        data_.PE_[size_t(PE::hlt)] = axiom(normalize_PE<PE::hlt>, type, Tag::PE, flags_t(PE::hlt), {op2str(PE::hlt)});
        data_.PE_[size_t(PE::run)] = axiom(normalize_PE<PE::run>, type, Tag::PE, flags_t(PE::run), {op2str(PE::run)});
    } { // known: ΠT: *. ΠT. bool
        auto type = pi(kind())->set_domain(kind());
        auto T = type->param({"T"});
        type->set_codomain(pi(T, type_bool()));
        data_.PE_[size_t(PE::known)] = axiom(normalize_PE<PE::known>, type, Tag::PE, flags_t(PE::known), {op2str(PE::known)});
    } { // bit: Πw: nat. Π[«bool; bool», int w, int w]. int w
        auto type = pi(kind())->set_domain(nat);
        auto int_w = type_int(type->param({"w"}));
        type->set_codomain(pi({arr({2, 2}, type_bool()), int_w, int_w}, int_w));
        data_.op_bit_ = axiom(normalize_bit, type, Tag::Bit, 0, {"bit"});
    } { // bitcast: Π[D: *, S: *]. ΠS. D
        auto type = pi(kind())->set_domain({kind(), kind()});
        auto D = type->param(0, {"D"});
        auto S = type->param(1, {"S"});
        type->set_codomain(pi(S, D));
        data_.op_bitcast_ = axiom(normalize_bitcast, type, Tag::Bitcast, 0, {"bitcast"});
    } { // lea:, Π[n: nat, Ts: «n; *», as: nat]. Π[ptr(«j: n; Ts#j», as), i: int n]. ptr(Ts#i, as)
        auto domain = sigma(universe(), 3);
        domain->set(0, nat);
        domain->set(1, arr(domain->param(0, {"n"}), kind()));
        domain->set(2, nat);
        auto pi1 = pi(kind())->set_domain(domain);
        auto n  = pi1->param(0, {"n"});
        auto Ts = pi1->param(1, {"Ts"});
        auto as = pi1->param(2, {"as"});
        auto in = arr_nom(n);
        in->set(extract(Ts, in->param({"j"})));
        auto pi2 = pi(kind())->set_domain({type_ptr(in, as), type_int(n)});
        pi2->set_codomain(type_ptr(extract(Ts, pi2->param(1, {"i"})), as));
        pi1->set_codomain(pi2);
        data_.op_lea_ = axiom(normalize_lea, pi1, Tag::LEA, 0, {"lea"});
    } { // sizeof: ΠT: *. nat
        data_.op_sizeof_ = axiom(normalize_sizeof, pi(kind(), nat), Tag::Sizeof, 0, {"sizeof"});
    } { // load:  Π[T: *, as: nat]. Π[M, ptr(T, as)]. [M, T]
        auto type = pi(kind())->set_domain({kind(), nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi({mem, ptr}, sigma({mem, T})));
        data_.op_load_ = axiom(normalize_load, type, Tag::Load, 0, {"load"});
    } { // store: Π[T: *, as: nat]. Π[M, ptr(T, as), T]. M
        auto type = pi(kind())->set_domain({kind(), nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi({mem, ptr, T}, mem));
        data_.op_store_ = axiom(normalize_store, type, Tag::Store, 0, {"store"});
    } { // alloc: Π[T: *, as: nat]. ΠM. [M, ptr(T, as)]
        auto type = pi(kind())->set_domain({kind(), nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi(mem, sigma({mem, ptr})));
        data_.op_alloc_ = axiom(nullptr, type, Tag::Alloc, 0, {"alloc"});
    } { // slot: Π[T: *, as: nat]. Π[M, nat]. [M, ptr(T, as)]
        auto type = pi(kind())->set_domain({kind(), nat});
        auto T  = type->param(0, {"T"});
        auto as = type->param(1, {"as"});
        auto ptr = type_ptr(T, as);
        type->set_codomain(pi({mem, nat}, sigma({mem, ptr})));
        data_.op_slot_ = axiom(nullptr, type, Tag::Slot, 0, {"slot"});
    } { // type_tangent_vector: Π*. *
        data_.type_tangent_vector_ = axiom(normalize_tangent, pi(kind(), kind()), Tag::TangentVector, 0, {"tangent"});
    }  { // op_grad: Π[T: *, R: *]. Π(ΠT. R). ΠT. tangent T
        auto type = pi(kind())->set_domain({kind(), kind()});
        auto T = type->param(0, {"T"});
        auto R = type->param(1, {"R"});
        auto tangent_T = type_tangent_vector(T);
        type->set_codomain(pi(pi(T, R), pi(T, tangent_T)));
        data_.op_grad_ = axiom(nullptr, type, Tag::Grad, 0, {"∇"});
    }
}

World::~World() {
    for (auto def : data_.defs_) def->~Def();
}

/*
 * core calculus
 */

#if 0
// TODO use for sigma
static const Def* lub(const Def* t1, const Def* t2) {
    if (t1->isa<Universe>()) return t1;
    if (t2->isa<Universe>()) return t2;
    assert(t1->isa<Kind>() && t2->isa<Kind>());
    return t1;
}
#endif

const Pi* World::pi(const Def* domain, const Def* codomain, Debug dbg) {
    return unify<Pi>(2, codomain->type(), domain, codomain, debug(dbg));
}

const Lam* World::lam(const Def* domain, const Def* filter, const Def* body, Debug dbg) {
    auto p = pi(domain, body->type());
    return unify<Lam>(2, p, filter, body, debug(dbg));
}

const Def* World::app(const Def* callee, const Def* arg, Debug dbg) {
    auto pi = callee->type()->as<Pi>();

    if (err()) {
        if (!assignable(pi->domain(), arg)) err()->ill_typed_app(callee, arg);
    }

    auto type = pi->apply(arg).back();
    auto [axiom, currying_depth] = get_axiom(callee); // TODO move down again
    if (axiom && currying_depth == 1) {
        if (auto normalize = axiom->normalizer())
            return normalize(type, callee, arg, debug(dbg));
    }

    return unify<App>(2, axiom, currying_depth-1, type, callee, arg, debug(dbg));
}

const Def* World::raw_app(const Def* callee, const Def* arg, Debug dbg) {
    auto pi = callee->type()->as<Pi>();
    auto type = pi->apply(arg).back();
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
    auto sigma = infer_sigma(*this, ops);
    auto t = tuple(sigma, ops, dbg);
    if (err() && !assignable(sigma, t)) {
        assert(false && "TODO: error msg");
    }

    return t;
}

const Def* World::tuple(const Def* type, Defs ops, Debug dbg) {
    if (err()) {
    // TODO type-check type vs inferred type
    }

    auto n = ops.size();
    if (!type->isa_nominal<Sigma>()) {
        if (n == 0) return tuple();
        if (n == 1) return ops[0];
        if (std::all_of(ops.begin()+1, ops.end(), [&](auto op) { return ops[0] == op; })) return pack(n, ops[0]);
    }

    // eta rule for tuples:
    // (extract(tup, 0), extract(tup, 1), extract(tup, 2)) -> tup
    if (n != 0) if (auto extract = ops[0]->isa<Extract>()) {
        auto tup = extract->tuple();
        bool eta = tup->type() == type;
        for (size_t i = 0; i != n && eta; ++i) {
            if (auto extract = ops[i]->isa<Extract>()) {
                if (auto index = isa_lit(extract->index())) {
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

const Def* World::which(const Def* value, Debug dbg) {
    if (auto insert = value->isa<Insert>())
        return insert->index();
    return unify<Which>(1, value->type()->arity(), value, debug(dbg));
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
    if (trivial)
        return ptrns[0]->as<Ptrn>()->apply(arg).back();
    if (ptrns.size() == 1 && !trivial) {
        if (err()) err()->incomplete_match(match);
        return bot(type);
    }
    // Constant folding
    if (arg->is_const()) {
        for (auto ptrn : ptrns) {
            // If the pattern matches the argument
            if (ptrn->as<Ptrn>()->matches(arg))
                return ptrn->as<Ptrn>()->apply(arg).back();
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
            res |= as_lit(proj(proj(tuple, 2, a_flags & 1), 2, b_flags & 1)) << 31_u32;
        res >>= (31_u32 - u32(num_bits));

        auto& world = tuple->world();
        if constexpr (tag == Tag::RCmp)
            return world.op(RCmp(res), /*rmode*/ a_cmp->decurry()->arg(0), a_cmp->arg(0), a_cmp->arg(1), dbg);
        else
            return world.op(ICmp(res), a_cmp->arg(0), a_cmp->arg(1), dbg);
    }
    return nullptr;
}

const Def* World::extract(const Def* ex_type, const Def* tup, const Def* index, Debug dbg) {
    if (index->isa<Arr>() || index->isa<Pack>()) {
        Array<const Def*> ops(as_lit(index->arity()), [&](size_t) { return extract(tup, index->ops().back()); });
        return index->isa<Arr>() ? sigma(ops, dbg) : tuple(ops, dbg);
    } else if (index->isa<Sigma>() || index->isa<Tuple>()) {
        auto n = index->num_ops();
        Array<const Def*> idx(n, [&](size_t i) { return index->op(i); });
        Array<const Def*> ops(n, [&](size_t i) { return proj(tup, n, as_lit(idx[i])); });
        return index->isa<Sigma>() ? sigma(ops, dbg) : tuple(ops, dbg);
    }

    auto type = tup->type()->reduce();
    assertf(alpha_equiv(type->arity(), isa_sized_type(index->type())),
            "extracting from tuple '{}' of arity '{}' with index '{}' of type '{}'", tup, type->arity(), index, index->type());

    // nominal sigmas can be 1-tuples
    if (auto bound = isa_lit(isa_sized_type(index->type())); bound && *bound == 1 && !tup->type()->isa_nominal<Sigma>()) return tup;
    if (auto pack = tup->isa<Pack>()) return pack->body();

    // extract(insert(x, index, val), index) -> val
    if (auto insert = tup->isa<Insert>()) {
        if (index == insert->index())
            return insert->value();
    }

    if (auto i = isa_lit(index)) {
        if (auto tuple = tup->isa<Tuple>()) return tuple->op(*i);

        // extract(insert(x, j, val), i) -> extract(x, i) where i != j (guaranteed by rule above)
        if (auto insert = tup->isa<Insert>()) {
            if (insert->index()->isa<Lit>())
                return extract(insert->tuple(), index, dbg);
        }

        if (type->isa<Sigma>() || type->isa<Union>())
            return unify<Extract>(2, ex_type ? ex_type : type->op(*i), tup, index, debug(dbg));
    }

    if (auto arr = type->isa<Arr>()) {
        if (auto tuple = tup->isa<Tuple>()) {
            // TODO we could even deal with an offset
            // extract((0, 1, 2, ...), i) -> i
            bool ascending = true;
            for (size_t i = 0, e = tuple->num_ops(); i != e && ascending; ++i) {
                if (auto lit = isa_lit(tuple->op(i)))
                    ascending &= *lit == i;
                else
                    ascending = false;
            }

            if (ascending)
                return op_bitcast(arr->body(), index, dbg);

            // extract((a, b, c, ...), extract((..., 2, 1, 0), i)) -> extract(..., c, b, a), i
            // this also deals with NOT
            if (auto i_ex = index->isa<Extract>()) {
                if (auto i_tup = i_ex->tuple()->isa<Tuple>()) {
                    bool descending = true;
                    for (size_t i = 0, e = i_tup->num_ops(); i != e && descending; ++i) {
                        if (auto lit = isa_lit(i_tup->op(i)))
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
        auto inner_type = inner->tuple()->type()->reduce();
        auto arity = as_lit(inner_type->arity());

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

    type = type->as<Arr>()->body();
    return unify<Extract>(2, type, tup, index, debug(dbg));
}

const Def* World::insert(const Def* tup, const Def* index, const Def* val, Debug dbg) {
    auto type = tup->type()->reduce();
    assertf(alpha_equiv(type->arity(), isa_sized_type(index->type())),
            "inserting into tuple {} of arity {} with index {} of type {}", tup, type->arity(), index, index->type());

    if (auto bound = isa_lit(isa_sized_type(index->type())); bound && *bound == 1)
        return tuple(tup, {val}, dbg); // tup could be nominal - that's why the tuple ctor is needed

    // insert((a, b, c, d), 2, x) -> (a, b, x, d)
    if (auto t = tup->isa<Tuple>()) return t->refine(as_lit(index), val);

    // insert(‹4; x›, 2, y) -> (x, x, y, x)
    if (auto pack = tup->isa<Pack>()) {
        if (auto a = isa_lit(pack->arity())) {
            Array<const Def*> new_ops(*a, pack->body());
            new_ops[as_lit(index)] = val;
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

bool is_shape(const Def* s) {
    if (s->type()->isa<Nat>()) return true;
    if (auto tup  = s->isa<Tuple>()) return std::all_of(tup->ops().begin(), tup->ops().end(), is_shape);
    if (auto pack = s->isa<Pack >()) return is_shape(pack->body());

    return false;
}

const Def* World::arr(const Def* shape, const Def* body, Debug dbg) {
    assert(is_shape(shape));

    if (auto a = isa_lit<u64>(shape)) {
        if (*a == 0) return sigma();
        if (*a == 1) return body;
    }

    return unify<Arr>(2, kind(), shape, body, debug(dbg));
}

const Def* World::pack(const Def* shape, const Def* body, Debug dbg) {
    assert(is_shape(shape));

    if (auto a = isa_lit<u64>(shape)) {
        if (*a == 0) return tuple();
        if (*a == 1) return body;
    }

    auto type = arr(shape, body->type());
    return unify<Pack>(1, type, body, debug(dbg));
}

const Def* World::arr(Defs shape, const Def* body, Debug dbg) {
    if (shape.empty()) return body;
    return arr(shape.skip_back(), arr(shape.back(), body, dbg), dbg);
}

const Def* World::pack(Defs shape, const Def* body, Debug dbg) {
    if (shape.empty()) return body;
    return pack(shape.skip_back(), pack(shape.back(), body, dbg), dbg);
}

const Lit* World::lit_int(const Def* type, u64 i, Debug dbg) {
    auto size = isa_sized_type(type);
    if (size->isa<Top>()) return lit(size, i, dbg);

    if (auto a = isa_lit(size)) {
        if (err() && *a != 0 && i >= *a) err()->index_out_of_range(*a, i);
    }

    return lit(type, i, dbg);
}

const Def* World::bot_top(bool is_top, const Def* type, Debug dbg) {
    if (auto arr = type->isa<Arr>()) return pack(arr->shape(), bot_top(is_top, arr->body()), dbg);
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
    if (auto arr   = t->isa<Arr  >()) return world.pack(arr->shape(), arr->body());
    return t;
}

const Def* World::op_lea(const Def* ptr, const Def* index, Debug dbg) {
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();
    auto Ts = tuple_of_types(pointee);
    return app(app(op_lea(), {pointee->arity(), Ts, addr_space}), {ptr, index}, debug(dbg));
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

const Def* World::gid2def(u32 gid) {
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
    unique_stack<DefSet> defs;

    auto push = [&](const Def* def) {
        if (!def->is_const()) {
            if (auto nom = def->isa_nominal())
                noms.push(nom);
            else
                defs.push(def);
        }
    };

    for (const auto& [name, nom] : externals()) {
        assert(nom->is_set() && "external must not be empty");
        noms.push(nom);
    }

    while (!noms.empty()) {
        auto nom = noms.pop();
        if (elide_empty && !nom->is_set()) continue;
        Scope scope(nom);
        f(scope);
        scope.visit({}, {}, {}, {}, [&](const Def* def) { push(def); });

        while (!defs.empty()) {
            for (auto op : defs.pop()->extended_ops()) push(op);
        }
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

const Def* World::op_grad(const Def* fn, Debug dbg) {
    if (fn->type()->isa<Pi>()) {
        auto ds_fn = cps2ds(fn);
        auto ds_pi = ds_fn->type()->as<Pi>();
        auto to_grad = app(data_.op_grad_, {ds_pi->domain(), ds_pi->codomain()}, dbg);
        auto grad = app(to_grad, ds_fn, dbg);
        return ds2cps(grad);
    }

    THORIN_UNREACHABLE;
}

const Def* World::type_tangent_vector(const Def* primal_type, Debug dbg) {
    return app(data_.type_tangent_vector_, primal_type, dbg);
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
std::string World::colorize(const std::string& str, int) {
#endif
    return str;
}

void World::set(std::unique_ptr<ErrorHandler>&& err) { err_ = std::move(err); }

template void Streamable<World>::write(const std::string& filename) const;
template void Streamable<World>::write() const;
template void Streamable<World>::dump() const;
template void World::visit<true> (VisitFn) const;
template void World::visit<false>(VisitFn) const;

}
