#include "thorin/def.h"
#include "thorin/world.h"

// TODO rewrite int normalization to work on all int modulos.
// This would also remove a lot of template magic.

namespace thorin {

template<class O> constexpr bool is_int      () { return true;  }
template<>        constexpr bool is_int<ROp >() { return false; }
template<>        constexpr bool is_int<RCmp>() { return false; }

/*
 * small helpers
 */

#if 0
static const Def* is_not(const Def* def) {
    if (auto ixor = isa<Tag::Bit>(Bit::_xor, def)) {
        auto [x, y] = ixor->args<2>();
        if (auto lit = isa_lit(x); lit && lit == as_lit(isa_sized_type(x->type())) - 1_u64) return y;
    }
    return nullptr;
}
#endif

template<class T> static T get(u64 u) { return bitcast<T>(u); }

/// Use like this:
/// @code a op b = tab[a][b] @endcode
constexpr std::array<std::array<uint64_t, 2>, 2> make_truth_table(Bit op) {
    return {{ {tag_t(op) & tag_t(0b0001) ? u64(-1) : 0, tag_t(op) & tag_t(0b0100) ? u64(-1) : 0},
              {tag_t(op) & tag_t(0b0010) ? u64(-1) : 0, tag_t(op) & tag_t(0b1000) ? u64(-1) : 0} }};
}

template<class T> constexpr bool is_commutative(T) { return false; }
constexpr bool is_commutative(Wrap op) { return op == Wrap:: add || op == Wrap::mul; }
constexpr bool is_commutative(ROp  op) { return op == ROp :: add || op == ROp ::mul; }
constexpr bool is_commutative(ICmp op) { return op == ICmp::   e || op == ICmp:: ne; }
constexpr bool is_commutative(RCmp op) { return op == RCmp::   e || op == RCmp:: ne; }
constexpr bool is_commutative(Bit  op) {
    auto tab = make_truth_table(op);
    return tab[0][1] == tab[1][0];
}

template<class T> constexpr bool is_associative(T op) { return is_commutative(op); }
constexpr bool is_associative(Bit op) {
    switch (op) {
        case Bit::   t:
        case Bit::_xor:
        case Bit::_and:
        case Bit::nxor:
        case Bit::   a:
        case Bit::   b:
        case Bit:: _or:
        case Bit::   f: return true;
        default       : return false;
    }
}

/*
 * Fold
 */

// This code assumes two-complement arithmetic for unsigned operations.
// This is *implementation-defined* but *NOT* *undefined behavior*.

class Res {
public:
    Res()
        : data_{}
    {}
    template<class T>
    Res(T val)
        : data_(bitcast<u64>(val))
    {}

    constexpr const u64& operator*() const& { return *data_; }
    constexpr u64& operator*() & { return *data_; }
    explicit operator bool() const { return data_.has_value(); }

private:
    std::optional<u64> data_;
};

template<class T, T, nat_t> struct Fold {};

template<nat_t w> struct Fold<Wrap, Wrap::add, w> {
    static Res run(u64 a, u64 b, bool /*nsw*/, bool nuw) {
        auto x = get<w2u<w>>(a), y = get<w2u<w>>(b);
        decltype(x) res = x + y;
        if (nuw && res < x) return {};
        // TODO nsw
        return res;
    }
};

template<nat_t w> struct Fold<Wrap, Wrap::sub, w> {
    static Res run(u64 a, u64 b, bool /*nsw*/, bool /*nuw*/) {
        using UT = w2u<w>;
        auto x = get<UT>(a), y = get<UT>(b);
        decltype(x) res = x - y;
        //if (nuw && y && x > std::numeric_limits<UT>::max() / y) return {};
        // TODO nsw
        return res;
    }
};

template<nat_t w> struct Fold<Wrap, Wrap::mul, w> {
    static Res run(u64 a, u64 b, bool /*nsw*/, bool /*nuw*/) {
        using UT = w2u<w>;
        auto x = get<UT>(a), y = get<UT>(b);
        if constexpr (std::is_same_v<UT, bool>)
            return UT(x & y);
        else
            return UT(x * y);
        // TODO nsw/nuw
    }
};

template<nat_t w> struct Fold<Wrap, Wrap::shl, w> {
    static Res run(u64 a, u64 b, bool nsw, bool nuw) {
        using T = w2u<w>;
        auto x = get<T>(a), y = get<T>(b);
        if (u64(y) >= w) return {};
        decltype(x) res;
        if constexpr (std::is_same_v<T, bool>)
            res = bool(u64(x) << u64(y));
        else
            res = x << y;
        if (nuw && res < x) return {};
        if (nsw && get_sign(x) != get_sign(res)) return {};
        return res;
    }
};

template<nat_t w> struct Fold<Div, Div::sdiv, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } };
template<nat_t w> struct Fold<Div, Div::udiv, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } };
template<nat_t w> struct Fold<Div, Div::srem, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } };
template<nat_t w> struct Fold<Div, Div::urem, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } };

template<nat_t w> struct Fold<Shr, Shr::ashr, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } };
template<nat_t w> struct Fold<Shr, Shr::lshr, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } };

template<nat_t w> struct Fold<ROp, ROp:: add, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) + get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: sub, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) - get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: mul, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) * get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: div, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) / get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: rem, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(rem(get<T>(a), get<T>(b))); } };

template<ICmp cmp, nat_t w> struct Fold<ICmp, cmp, w> {
    inline static Res run(u64 a, u64 b) {
        using T = w2u<w>;
        auto x = get<T>(a), y = get<T>(b);
        bool result = false;
        auto pm = !(x >> T(w-1)) &&  (y >> T(w-1));
        auto mp =  (x >> T(w-1)) && !(y >> T(w-1));
        result |= ((cmp & ICmp::_x) != ICmp::_f) && pm;
        result |= ((cmp & ICmp::_y) != ICmp::_f) && mp;
        result |= ((cmp & ICmp::_g) != ICmp::_f) && x > y && !mp;
        result |= ((cmp & ICmp::_l) != ICmp::_f) && x < y && !pm;
        result |= ((cmp & ICmp:: e) != ICmp::_f) && x == y;
        return result;
    }
};

template<RCmp cmp, nat_t w> struct Fold<RCmp, cmp, w> {
    inline static Res run(u64 a, u64 b) {
        using T = w2r<w>;
        auto x = get<T>(a), y = get<T>(b);
        bool result = false;
        result |= ((cmp & RCmp::u) != RCmp::f) && std::isunordered(x, y);
        result |= ((cmp & RCmp::g) != RCmp::f) && x > y;
        result |= ((cmp & RCmp::l) != RCmp::f) && x < y;
        result |= ((cmp & RCmp::e) != RCmp::f) && x == y;
        return result;
    }
};

template<Conv op, nat_t, nat_t> struct FoldConv {};
template<nat_t dw, nat_t sw> struct FoldConv<Conv::s2s, dw, sw> { static Res run(u64 src) { return w2s<dw>(get<w2s<sw>>(src)); } };
template<nat_t dw, nat_t sw> struct FoldConv<Conv::u2u, dw, sw> { static Res run(u64 src) { return w2u<dw>(get<w2u<sw>>(src)); } };
template<nat_t dw, nat_t sw> struct FoldConv<Conv::s2r, dw, sw> { static Res run(u64 src) { return w2r<dw>(get<w2s<sw>>(src)); } };
template<nat_t dw, nat_t sw> struct FoldConv<Conv::u2r, dw, sw> { static Res run(u64 src) { return w2r<dw>(get<w2u<sw>>(src)); } };
template<nat_t dw, nat_t sw> struct FoldConv<Conv::r2s, dw, sw> { static Res run(u64 src) { return w2s<dw>(get<w2r<sw>>(src)); } };
template<nat_t dw, nat_t sw> struct FoldConv<Conv::r2u, dw, sw> { static Res run(u64 src) { return w2u<dw>(get<w2r<sw>>(src)); } };
template<nat_t dw, nat_t sw> struct FoldConv<Conv::r2r, dw, sw> { static Res run(u64 src) { return w2r<dw>(get<w2r<sw>>(src)); } };

/*
 * bigger logic used by several ops
 */

template<class O>
static void commute(O op, const Def*& a, const Def*& b) {
    if (is_commutative(op)) {
        if (b->isa<Lit>() || (a->gid() > b->gid() && !a->isa<Lit>())) // swap lit to left, or smaller gid to left if no lit present
            std::swap(a, b);
    }
}

/**
 * Reassociates @p a und @p b according to following rules.
 * We use the following naming convention while literals are prefixed with an 'l':
@verbatim
    a    op     b
(x op y) op (z op w)

(1)     la    op (lz op w) -> (la op lz) op w
(2) (lx op y) op (lz op w) -> (lx op lz) op (y op w)
(3)      a    op (lz op w) ->  lz op (a op w)
(4) (lx op y) op      b    ->  lx op (y op b)
@endverbatim
 */
template<tag_t tag>
static const Def* reassociate(Tag2Enum<tag> op, World& world, [[maybe_unused]] const App* ab, const Def* a, const Def* b, const Def* dbg) {
    if (!is_associative(op)) return nullptr;

    auto la = a->isa<Lit>();
    auto xy = isa<tag>(op, a);
    auto zw = isa<tag>(op, b);
    auto lx = xy ? xy->arg(0)->template isa<Lit>() : nullptr;
    auto lz = zw ? zw->arg(0)->template isa<Lit>() : nullptr;
    auto  y = xy ? xy->arg(1) : nullptr;
    auto  w = zw ? zw->arg(1) : nullptr;

    std::function<const Def*(const Def*, const Def*)> make_op;

    if constexpr (tag == Tag::ROp) {
        // build rmode for all new ops by using the least upper bound of all involved apps
        nat_t rmode = RMode::bot;
        auto check_mode = [&](const App* app) {
            auto app_m = isa_lit(app->arg(0));
            if (!app_m || !has(*app_m, RMode::reassoc)) return false;
            rmode &= *app_m; // least upper bound
            return true;
        };

        if (!check_mode(ab)) return nullptr;
        if (lx && !check_mode(xy->decurry())) return nullptr;
        if (lz && !check_mode(zw->decurry())) return nullptr;

        make_op = [&](const Def* a, const Def* b) { return world.op(op, rmode, a, b, dbg); };
    } else if constexpr (tag == Tag::Wrap) {
        // if we reassociate Wraps, we have to forget about nsw/nuw
        make_op = [&](const Def* a, const Def* b) { return world.op(op, WMode::none, a, b, dbg); };
    } else {
        make_op = [&](const Def* a, const Def* b) { return world.op(op, a, b, dbg); };
    }

    if (la && lz) return make_op(make_op(la, lz), w);             // (1)
    if (lx && lz) return make_op(make_op(lx, lz), make_op(y, w)); // (2)
    if (lz)       return make_op(lz, make_op(a, w));              // (3)
    if (lx)       return make_op(lx, make_op(y, b));              // (4)

    return nullptr;
}

/// @attention Note that @p a and @p b are passed by reference as fold also commutes if possible. See commute().
template<class Op, Op op>
static const Def* fold(World& world, const Def* type, const App* callee, const Def*& a, const Def*& b, const Def* dbg) {
    static constexpr int min_w = std::is_same_v<Op, ROp> || std::is_same_v<Op, RCmp> ? 16 : 1;
    auto la = a->isa<Lit>(), lb = b->isa<Lit>();

    if (a->isa<Bot>() || b->isa<Bot>()) return world.bot(type, dbg);

    if (la && lb) {
        nat_t width;
        [[maybe_unused]] bool nsw = false, nuw = false;
        if constexpr (std::is_same_v<Op, Wrap>) {
            auto [mode, w] = callee->args<2>(as_lit<nat_t>);
            nsw = mode & WMode::nsw;
            nuw = mode & WMode::nuw;
            width = w;
        } else {
            width = as_lit(a->type()->as<App>()->arg());
        }

        if (is_int<Op>()) width = *mod2width(width);

        Res res;
        switch (width) {
#define CODE(i)                                                                     \
            case i:                                                                 \
                if constexpr (i >= min_w) {                                         \
                    if constexpr (std::is_same_v<Op, Wrap>)                         \
                        res = Fold<Op, op, i>::run(la->get(), lb->get(), nsw, nuw); \
                    else                                                            \
                        res = Fold<Op, op, i>::run(la->get(), lb->get());           \
                }                                                                   \
                break;
            THORIN_1_8_16_32_64(CODE)
#undef CODE
            default: THORIN_UNREACHABLE;
        }

        return res ? world.lit(type, *res, dbg) : world.bot(type, dbg);
    }

    commute(op, a, b);
    return nullptr;
}

/*
 * normalize
 */

template<tag_t tag>
static const Def* merge_cmps(std::array<std::array<uint64_t, 2>, 2> tab, const Def* a, const Def* b, const Def* dbg) {
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
            res |= tab[a_flags & 1][b_flags & 1] << 31_u32;
        res >>= (31_u32 - u32(num_bits));

        auto& world = a->world();
        if constexpr (tag == Tag::RCmp)
            return world.op(RCmp(res), /*rmode*/ a_cmp->decurry()->arg(0), a_cmp->arg(0), a_cmp->arg(1), dbg);
        else
            return world.op(ICmp(res), a_cmp->arg(0), a_cmp->arg(1), dbg);
    }

    return nullptr;
}

template<Bit op>
const Def* normalize_Bit(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->outs<2>();
    auto w = isa_lit(callee->arg());

    commute(op, a, b);

    auto tab = make_truth_table(op);
    if (auto res = merge_cmps<Tag::ICmp>(tab, a, b, dbg)) return res;
    if (auto res = merge_cmps<Tag::RCmp>(tab, a, b, dbg)) return res;

    auto la = isa_lit(a);
    auto lb = isa_lit(b);

    switch (op) {
        case Bit::    f: return world.lit_int(*w,        0);
        case Bit::    t: return world.lit_int(*w, *w-1_u64);
        case Bit::    a: return a;
        case Bit::    b: return b;
        case Bit::   na: return world.op_negate(a, dbg);
        case Bit::   nb: return world.op_negate(b, dbg);
        case Bit:: ciff: return world.op(Bit:: iff, b, a, dbg);
        case Bit::nciff: return world.op(Bit::niff, b, a, dbg);
        default:         break;
    }

    if (la && lb) {
        switch (op) {
            case Bit::_and: return world.lit_int    (*w,   *la &  *lb);
            case Bit:: _or: return world.lit_int    (*w,   *la |  *lb);
            case Bit::_xor: return world.lit_int    (*w,   *la ^  *lb);
            case Bit::nand: return world.lit_int_mod(*w, ~(*la &  *lb));
            case Bit:: nor: return world.lit_int_mod(*w, ~(*la |  *lb));
            case Bit::nxor: return world.lit_int_mod(*w, ~(*la ^  *lb));
            case Bit:: iff: return world.lit_int_mod(*w, ~ *la |  *lb);
            case Bit::niff: return world.lit_int    (*w,   *la & ~*lb);
            default: THORIN_UNREACHABLE;
        }
    }

    auto unary = [&](bool x, bool y, const Def* a) -> const Def* {
        if (!x && !y) return world.lit_int(*w, 0);
        if ( x &&  y) return world.lit_int(*w, *w-1_u64);
        if (!x &&  y) return a;
        if ( x && !y && op != Bit::_xor) return world.op_negate(a, dbg);
        return nullptr;
    };

    if (is_commutative(op) && a == b) {
        if (auto res = unary(tab[0][0], tab[1][1], a)) return res;
    }

    if (la) {
        if (*la == 0) {
            if (auto res = unary(tab[0][0], tab[0][1], b)) return res;
        } else if (*la == *w-1_u64) {
            if (auto res = unary(tab[1][0], tab[1][1], b)) return res;
        }
    }

    if (lb) {
        if (*lb == 0) {
            if (auto res = unary(tab[0][0], tab[1][0], b)) return res;
        } else if (*lb == *w-1_u64) {
            if (auto res = unary(tab[0][1], tab[1][1], b)) return res;
        }
    }

#if 0
    // TODO
    // commute NOT to b
    if (is_commutative(op) && is_not(a)) std::swap(a, b);

    if (auto bb = is_not(b); bb && a == bb) {
        switch (op) {
            case IOp::iand: return world.lit_int(*w,       0);
            case IOp::ior : return world.lit_int(*w, *w-1_u64);
            case IOp::ixor: return world.lit_int(*w, *w-1_u64);
            default: THORIN_UNREACHABLE;
        }
    }

    auto absorption = [&](IOp op1, IOp op2) -> const Def* {
        if (op == op1) {
            if (auto xy = isa<Tag::IOp>(op2, a)) {
                auto [x, y] = xy->args<2>();
                if (x == b) return y; // (b op2 y) op1 b -> y
                if (y == b) return x; // (x op2 b) op1 b -> x
            }

            if (auto zw = isa<Tag::IOp>(op2, b)) {
                auto [z, w] = zw->args<2>();
                if (z == a) return w; // a op1 (a op2 w) -> w
                if (w == a) return z; // a op1 (z op2 a) -> z
            }
        }
        return nullptr;
    };

    auto simplify1 = [&](IOp op1, IOp op2) -> const Def* { // AFAIK this guy has no name
        if (op == op1) {
            if (auto xy = isa<Tag::IOp>(op2, a)) {
                auto [x, y] = xy->args<2>();
                if (auto yy = is_not(y); yy && yy == b) return world.op(op1, x, b, dbg); // (x op2 not b) op1 b -> x op1 y
            }

            if (auto zw = isa<Tag::IOp>(op2, b)) {
                auto [z, w] = zw->args<2>();
                if (auto ww = is_not(w); ww && ww == a) return world.op(op1, a, z, dbg); // a op1 (z op2 not a) -> a op1 z
            }
        }
        return nullptr;
    };

    auto simplify2 = [&](IOp op1, IOp op2) -> const Def* { // AFAIK this guy has no name
        if (op == op1) {
            if (auto xy = isa<Tag::IOp>(op2, a)) {
                if (auto zw = isa<Tag::IOp>(op2, b)) {
                    auto [x, y] = xy->args<2>();
                    auto [z, w] = zw->args<2>();
                    if (auto yy = is_not(y); yy && x == z && yy == w) return x; // (z op2 not w) op1 (z op2 w) -> x
                    if (auto ww = is_not(w); ww && x == z && ww == y) return x; // (x op2 y) op1 (x op2 not y) -> x
                }
            }

        }
        return nullptr;
    };

    if (auto res = absorption(IOp::ior , IOp::iand)) return res;
    if (auto res = absorption(IOp::iand, IOp::ior )) return res;
    if (auto res = simplify1 (IOp::ior , IOp::iand)) return res;
    if (auto res = simplify1 (IOp::iand, IOp::ior )) return res;
    if (auto res = simplify2 (IOp::ior , IOp::iand)) return res;
    if (auto res = simplify2 (IOp::iand, IOp::ior )) return res;
#endif
    if (auto res = reassociate<Tag::Bit>(op, world, callee, a, b, dbg)) return res;

    return world.raw_app(callee, {a, b}, dbg);
}

template<Shr op>
const Def* normalize_Shr(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->outs<2>();
    auto w = isa_lit(callee->arg());

    if (auto result = fold<Shr, op>(world, type, callee, a, b, dbg)) return result;

    if (auto la = a->isa<Lit>()) {
        if (la == world.lit_int(*w, 0)) {
            switch (op) {
                case Shr::ashr: return la;
                case Shr::lshr: return la;
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (auto lb = b->isa<Lit>()) {
        if (lb == world.lit_int(*w, 0)) {
            switch (op) {
                case Shr::ashr: return a;
                case Shr::lshr: return a;
                default: THORIN_UNREACHABLE;
            }
        }

        if (lb->get() > *w) return world.bot(type, dbg);
    }

    return world.raw_app(callee, {a, b}, dbg);
}

template<Wrap op>
const Def* normalize_Wrap(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->outs<2>();
    auto [m, w] = callee->args<2>(isa_lit<nat_t>); // mode and width

    if (auto result = fold<Wrap, op>(world, type, callee, a, b, dbg)) return result;

    if (auto la = a->isa<Lit>()) {
        if (la == world.lit_int(*w, 0)) {
            switch (op) {
                case Wrap::add: return b;    // 0  + b -> b
                case Wrap::sub: break;
                case Wrap::mul: return la;   // 0  * b -> 0
                case Wrap::shl: return la;   // 0 << b -> 0
                default: THORIN_UNREACHABLE;
            }
        }

        if (la == world.lit_int(*w, 1)) {
            switch (op) {
                case Wrap::add: break;
                case Wrap::sub: break;
                case Wrap::mul: return b;    // 1  * b -> b
                case Wrap::shl: break;
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (auto lb = b->isa<Lit>()) {
        if (lb == world.lit_int(*w, 0)) {
            switch (op) {
                case Wrap::sub: return a;    // a  - 0 -> a
                case Wrap::shl: return a;    // a >> 0 -> a
                default: THORIN_UNREACHABLE;
                // add, mul are commutative, the literal has been normalized to the left
            }
        }

        if (op == Wrap::sub)
            return world.op(Wrap::add, *m, a, world.lit_int_mod(*w, ~lb->get() + 1_u64)); // a - lb -> a + (~lb + 1)
        else if (op == Wrap::shl && lb->get() > *w)
            return world.bot(type, dbg);
    }

    if (a == b) {
        switch (op) {
            case Wrap::add: return world.op(Wrap::mul, *m, world.lit_int(*w, 2), a, dbg); // a + a -> 2 * a
            case Wrap::sub: return world.lit_int(*w, 0);                                 // a - a -> 0
            case Wrap::mul: break;
            case Wrap::shl: break;
            default: THORIN_UNREACHABLE;
        }
    }

    if (auto res = reassociate<Tag::Wrap>(op, world, callee, a, b, dbg)) return res;

    return world.raw_app(callee, {a, b}, dbg);
}

template<Div op>
const Def* normalize_Div(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [mem, a, b] = arg->outs<3>();
    auto w = isa_lit(callee->arg());
    type = type->as<Sigma>()->op(1); // peel of actual type
    auto make_res = [&, mem = mem](const Def* res) { return world.tuple({mem, res}, dbg); };

    if (auto result = fold<Div, op>(world, type, callee, a, b, dbg)) return make_res(result);

    if (auto la = a->isa<Lit>()) {
        if (la == world.lit_int(*w, 0)) return make_res(la); // 0 / b -> 0 and 0 % b -> 0
    }

    if (auto lb = b->isa<Lit>()) {
        if (lb == world.lit_int(*w, 0)) return make_res(world.bot(type)); // a / 0 -> ⊥ and a % 0 -> ⊥

        if (lb == world.lit_int(*w, 1)) {
            switch (op) {
                case Div::sdiv: return make_res(a);                    // a / 1 -> a
                case Div::udiv: return make_res(a);                    // a / 1 -> a
                case Div::srem: return make_res(world.lit_int(*w, 0)); // a % 1 -> 0
                case Div::urem: return make_res(world.lit_int(*w, 0)); // a % 1 -> 0
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (a == b) {
        switch (op) {
            case Div::sdiv: return make_res(world.lit_int(*w, 1)); // a / a -> 1
            case Div::udiv: return make_res(world.lit_int(*w, 1)); // a / a -> 1
            case Div::srem: return make_res(world.lit_int(*w, 0)); // a % a -> 0
            case Div::urem: return make_res(world.lit_int(*w, 0)); // a % a -> 0
            default: THORIN_UNREACHABLE;
        }
    }

    return world.raw_app(callee, {mem, a, b}, dbg);
}

template<ROp op>
const Def* normalize_ROp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->outs<2>();
    auto [m, w] = callee->args<2>(isa_lit<nat_t>); // mode and width

    if (auto result = fold<ROp, op>(world, type, callee, a, b, dbg)) return result;

    // TODO check rmode properly
    if (m && *m == RMode::fast) {
        if (auto la = a->isa<Lit>()) {
            if (la == world.lit_real(*w, 0.0)) {
                switch (op) {
                    case ROp::add: return b;    // 0 + b -> b
                    case ROp::sub: break;
                    case ROp::mul: return la;   // 0 * b -> 0
                    case ROp::div: return la;   // 0 / b -> 0
                    case ROp::rem: return la;   // 0 % b -> 0
                    default: THORIN_UNREACHABLE;
                }
            }

            if (la == world.lit_real(*w, 1.0)) {
                switch (op) {
                    case ROp::add: break;
                    case ROp::sub: break;
                    case ROp::mul: return b;    // 1 * b -> b
                    case ROp::div: break;
                    case ROp::rem: break;
                    default: THORIN_UNREACHABLE;
                }
            }
        }

        if (auto lb = b->isa<Lit>()) {
            if (lb == world.lit_real(*w, 0.0)) {
                switch (op) {
                    case ROp::sub: return a;    // a - 0 -> a
                    case ROp::div: break;
                    case ROp::rem: break;
                    default: THORIN_UNREACHABLE;
                    // add, mul are commutative, the literal has been normalized to the left
                }
            }
        }

        if (a == b) {
            switch (op) {
                case ROp::add: return world.op(ROp::mul, world.lit_real(*w, 2.0), a, dbg); // a + a -> 2 * a
                case ROp::sub: return world.lit_real(*w, 0.0);                             // a - a -> 0
                case ROp::mul: break;
                case ROp::div: return world.lit_real(*w, 1.0);                             // a / a -> 1
                case ROp::rem: break;
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (auto res = reassociate<Tag::ROp>(op, world, callee, a, b, dbg)) return res;

    return world.raw_app(callee, {a, b}, dbg);
}

template<ICmp op>
const Def* normalize_ICmp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->outs<2>();

    if (auto result = fold<ICmp, op>(world, type, callee, a, b, dbg)) return result;
    if (op == ICmp::_f) return world.lit_false();
    if (op == ICmp::_t) return world.lit_true();
    if (a == b) {
        if (op == ICmp:: e) return world.lit_true();
        if (op == ICmp::ne) return world.lit_false();
    }

    return world.raw_app(callee, {a, b}, dbg);
}

template<RCmp op>
const Def* normalize_RCmp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->outs<2>();

    if (auto result = fold<RCmp, op>(world, type, callee, a, b, dbg)) return result;
    if (op == RCmp::f) return world.lit_false();
    if (op == RCmp::t) return world.lit_true();

    return world.raw_app(callee, {a, b}, dbg);
}

// TODO this currently hard-codes x86_64 ABI
// TODO in contrast to C, we might want to give singleton types like 'int 1' or '[]' a size of 0 and simply nuke each and every occurance of these types in a later phase
// TODO Pi and others
template<Trait op>
const Def* normalize_Trait(const Def*, const Def* callee, const Def* type, const Def* dbg) {
    auto& world = type->world();

    if (auto ptr = isa<Tag::Ptr>(type)) {
        return world.lit_nat(8);
    } else if (auto int_ = isa<Tag::Int>(type)) {
        if (int_->type()->isa<Top>()) return world.lit_nat(8);
        if (auto w = isa_lit(int_->arg())) {
            if (*w == 0) return world.lit_nat(8);
            if (*w <= 0x0000'0000'0000'0100_u64) return world.lit_nat(1);
            if (*w <= 0x0000'0000'0001'0000_u64) return world.lit_nat(2);
            if (*w <= 0x0000'0001'0000'0000_u64) return world.lit_nat(4);
            return world.lit_nat(8);
        }
    } else if (auto real = isa<Tag::Real>(type)) {
        if (auto w = isa_lit(real->arg())) {
            switch (*w) {
                case 16: return world.lit_nat(2);
                case 32: return world.lit_nat(4);
                case 64: return world.lit_nat(8);
                default: THORIN_UNREACHABLE;
            }
        }
    } else if (type->isa<Sigma>() || type->isa<Meet>()) {
        u64 offset = 0;
        u64 align = 1;
        for (auto t : type->ops()) {
            auto a = isa_lit(world.op(Trait::align, t));
            auto s = isa_lit(world.op(Trait::size , t));
            if (!a || !s) goto out;

            align = std::max(align, *a);
            offset = pad(offset, *a) + *s;
        }

        offset = pad(offset, align);
        u64 size = std::max(1_u64, offset);

        switch (op) {
            case Trait::align: return world.lit_nat(align);
            case Trait::size:  return world.lit_nat(size );
        }
    } else if (auto arr = type->isa_structural<Arr>()) {
        auto align = world.op(Trait::align, arr->body());

        if constexpr (op == Trait::align) return align;

        auto a = isa_lit(align);
        auto s = isa_lit(world.op(Trait::size , arr->body()));

        if (auto shape = isa_lit(arr->shape()); shape && a && s) {
            u64 factor = std::max(*a, *s);
            return world.lit_nat(factor * *shape);
        }
    } else if (auto join = type->isa<Join>()) {
        if (auto sigma = join->convert()) return world.op(op, sigma, dbg);
    }

out:
    return world.raw_app(callee, type, dbg);
}

#define TABLE(m) m( 1,  1) m( 1,  8) m( 1, 16) m( 1, 32) m( 1, 64) \
                 m( 8,  1) m( 8,  8) m( 8, 16) m( 8, 32) m( 8, 64) \
                 m(16,  1) m(16,  8) m(16, 16) m(16, 32) m(16, 64) \
                 m(32,  1) m(32,  8) m(32, 16) m(32, 32) m(32, 64) \
                 m(64,  1) m(64,  8) m(64, 16) m(64, 32) m(64, 64)

template<nat_t min_sw, nat_t min_dw, Conv op>
static const Def* fold_Conv(const Def* dst_type, const App* callee, const Def* src, const Def* dbg) {
    auto& world = dst_type->world();
    if (src->isa<Bot>()) return world.bot(dst_type, dbg);

    auto [lit_dw, lit_sw] = callee->args<2>(isa_lit<nat_t>);
    auto lit_src = src->isa<Lit>();
    if (lit_src && lit_dw && lit_sw) {
        if (op == Conv::u2u) {
            if (*lit_dw == 0) return world.lit(dst_type, as_lit(lit_src));
            return world.lit(dst_type, as_lit(lit_src) % *lit_dw);
        }

        if (isa<Tag::Int>(src->type())) *lit_sw = *mod2width(*lit_sw);
        if (isa<Tag::Int>( dst_type  )) *lit_dw = *mod2width(*lit_dw);

        Res res;
#define CODE(sw, dw)                                             \
        else if (*lit_dw == dw && *lit_sw == sw) {               \
            if constexpr (dw >= min_dw && sw >= min_sw)          \
                res = FoldConv<op, dw, sw>::run(lit_src->get()); \
        }
        if (false) {} TABLE(CODE)
#undef CODE
        if (res) return world.lit(dst_type, *res, dbg);
        return world.bot(dst_type, dbg);
    }

    return nullptr;
}

template<Conv op>
const Def* normalize_Conv(const Def* dst_type, const Def* c, const Def* src, const Def* dbg) {
    auto& world = dst_type->world();
    auto callee = c->as<App>();

    static constexpr auto min_sw = op == Conv::r2s || op == Conv::r2u || op == Conv::r2r ? 16 : 1;
    static constexpr auto min_dw = op == Conv::s2r || op == Conv::u2r || op == Conv::r2r ? 16 : 1;
    if (auto result = fold_Conv<min_sw, min_dw, op>(dst_type, callee, src, dbg)) return result;

    auto [dw, sw] = callee->args<2>(isa_lit<nat_t>);
    if (sw == dw && dst_type == src->type()) return src;

    if constexpr (op == Conv::s2s) {
        if (sw && dw && *sw < *dw) return world.op(Conv::u2u, dst_type, src, dbg);
    }

    return world.raw_app(callee, src, dbg);
}

template<PE op>
const Def* normalize_PE(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();

    if constexpr (op == PE::known) {
        if (world.is_pe_done() || isa<Tag::PE>(PE::hlt, arg)) return world.lit_false();
        if (arg->no_dep()) return world.lit_true();
    } else {
        if (world.is_pe_done()) return arg;
    }

    return world.raw_app(callee, arg, dbg);
}

template<Acc op>
const Def* normalize_Acc(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    return type->world().raw_app(callee, arg, dbg);
}

const Def* normalize_bitcast(const Def* dst_type, const Def* callee, const Def* src, const Def* dbg) {
    auto& world = dst_type->world();

    if (src->isa<Bot>())         return world.bot(dst_type);
    if (src->type() == dst_type) return src;

    if (auto other = isa<Tag::Bitcast>(src))
        return other->arg()->type() == dst_type ? other->arg() : world.op_bitcast(dst_type, other->arg(), dbg);

    if (auto lit = src->isa<Lit>()) {
        if (dst_type->isa<Nat>()) return world.lit(dst_type, lit->get(), dbg);
        if (isa_sized_type(dst_type)) return world.lit(dst_type, lit->get(), dbg);
    }

    return world.raw_app(callee, src, dbg);
}

const Def* normalize_lea(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [ptr, index] = arg->outs<2>();
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();

    if (auto a = isa_lit(pointee->arity()); a && *a ==  1) return ptr;
    //TODO

    return world.raw_app(callee, {ptr, index}, dbg);
}

const Def* normalize_load(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [mem, ptr] = arg->outs<2>();
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();

    if (ptr->isa<Bot>()) return world.tuple({mem, world.bot(type->as<Sigma>()->op(1))}, dbg);

    // loading an empty tuple can only result in an empty tuple
    if (auto sigma = pointee->isa<Sigma>(); sigma && sigma->num_ops() == 0)
        return world.tuple({mem, world.tuple(sigma->type(), {}, dbg)});

    return world.raw_app(callee, {mem, ptr}, dbg);
}

const Def* normalize_remem(const Def* type, const Def* callee, const Def* mem, const Def* dbg) {
    auto& world = type->world();

    //if (auto m = isa<Tag::Remem>(mem)) mem = m;
    return world.raw_app(callee, mem, dbg);
}

const Def* normalize_store(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [mem, ptr, val] = arg->outs<3>();

    if (ptr->isa<Bot>() || val->isa<Bot>()) return mem;
    if (auto pack = val->isa<Pack>(); pack && pack->body()->isa<Bot>()) return mem;
    if (auto tuple = val->isa<Tuple>()) {
        if (std::all_of(tuple->ops().begin(), tuple->ops().end(), [&](const Def* op) { return op->isa<Bot>(); }))
            return mem;
    }

    return world.raw_app(callee, {mem, ptr, val}, dbg);
}

static const Def* tangent_vector_type(const Def* primal_type) {
    auto& world = primal_type->world();

    if (isa<Tag::Real>(primal_type)) {
        return primal_type;
    }

    if (auto arr = primal_type->isa<Arr>()) {
        auto elem_tangent_type = world.type_tangent_vector(arr->op(1));

        // Array of non-differentiable elements is non-differentiable
        if (auto sigma = elem_tangent_type->isa<Sigma>(); sigma && sigma->num_ops() == 0) {
            return world.sigma();
        }

        return world.arr(arr->op(0), elem_tangent_type);
    }

    if (auto sigma = primal_type->isa<Sigma>()) {
        auto num_ops = sigma->num_ops();

        // Σs with a mem are function vars.
        if (auto mem = isa<Tag::Mem>(sigma->op(0))) {
            auto vars = (num_ops > 2) ? world.sigma(sigma->ops().skip_front()) : sigma->op(1);
            return world.sigma({mem, world.type_tangent_vector(vars)});
        }

        DefArray tangent_vectors(num_ops);
        for (size_t i = 0; i < num_ops; ++i) {
            tangent_vectors[i] = world.type_tangent_vector(sigma->op(i));
        }
        return world.sigma(tangent_vectors);
    }

    // Either non-differentiable or needs inlining.
    return nullptr;
}

const Def* normalize_tangent(const Def*, const Def* callee, const Def* arg, const Def* dbg) {
    if (auto tangent_vector = tangent_vector_type(arg)) {
        return tangent_vector;
    }

    // Needs more inlining.
    return arg->world().raw_app(callee, arg, dbg);
}

const Def* normalize_lift(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& w = type->world();
    auto callee = c->as<App>();
    auto is_os = callee->arg();
    auto [n_i, Is, n_o, Os, f] = is_os->outs<5>();
    auto [r, s] = callee->decurry()->args<2>();
    auto lr = isa_lit(r);
    auto ls = isa_lit(s);

    // TODO commute
    // TODO reassociate
    // TODO more than one Os
    // TODO select which Is/Os to lift

    if (lr && ls && *lr == 1 && *ls == 1) return w.app(f, arg, dbg);

    if (auto l_in = isa_lit(n_i)) {
        auto args = arg->outs(*l_in);

        if (lr && std::all_of(args.begin(), args.end(), [&](const Def* arg) { return is_tuple_or_pack(arg); })) {
            auto shapes = s->outs(*lr);
            auto s_n = isa_lit(shapes.front());

            if (s_n) {
                DefArray elems(*s_n, [&, f = f](size_t s_i) {
                    DefArray inner_args(args.size(), [&](size_t i) { return proj(args[i], *s_n, s_i); });
                    if (*lr == 1)
                        return w.app(f, inner_args);
                    else
                        return w.app(w.app(w.app(w.ax_lift(), {w.lit_nat(*lr - 1), w.tuple(shapes.skip_front())}), is_os), inner_args);
                });
                return w.tuple(elems);
            }
        }
    }

    return w.raw_app(callee, arg, dbg);
}

/*
 * instantiate templates
 */

#define CODE(T, o) template const Def* normalize_ ## T<T::o>(const Def*, const Def*, const Def*, const Def*);
THORIN_BIT  (CODE)
THORIN_SHR  (CODE)
THORIN_WRAP (CODE)
THORIN_DIV  (CODE)
THORIN_R_OP (CODE)
THORIN_I_CMP(CODE)
THORIN_R_CMP(CODE)
THORIN_TRAIT(CODE)
THORIN_CONV (CODE)
THORIN_PE   (CODE)
THORIN_ACC  (CODE)
#undef CODE

}
