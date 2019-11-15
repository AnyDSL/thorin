#include "thorin/def.h"
#include "thorin/util.h"
#include "thorin/world.h"

namespace thorin {

/*
 * small helpers
 */

#if 0
static const Def* is_not(const Def* def) {
    if (auto ixor = isa<Tag::IOp>(IOp::ixor, def)) {
        auto [x, y] = ixor->args<2>();
        if (auto lit = x->isa<Lit>(); lit && lit == def->world().lit_int(*get_width(lit->type()), u64(-1))) return y;
    }
    return nullptr;
}
#endif

template<class T> static T get(u64 u) { return bitcast<T>(u); }

template<class T> static bool is_commutative(T) { return false; }
static bool is_commutative(WOp  op) { return op == WOp :: add || op == WOp ::mul; }
static bool is_commutative(ROp  op) { return op == ROp :: add || op == ROp ::mul; }
static bool is_commutative(ICmp op) { return op == ICmp::   e || op == ICmp:: ne; }
static bool is_commutative(RCmp op) { return op == RCmp::   e || op == RCmp:: ne; }

template<class T> static bool is_associative(T op) { return is_commutative(op); }

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

template<nat_t w> struct Fold<WOp, WOp::add, w> {
    static Res run(u64 a, u64 b, bool /*nsw*/, bool nuw) {
        auto x = get<w2u<w>>(a), y = get<w2u<w>>(b);
        decltype(x) res = x + y;
        if (nuw && res < x) return {};
        // TODO nsw
        return res;
    }
};

template<nat_t w> struct Fold<WOp, WOp::sub, w> {
    static Res run(u64 a, u64 b, bool /*nsw*/, bool /*nuw*/) {
        using UT = w2u<w>;
        auto x = get<UT>(a), y = get<UT>(b);
        decltype(x) res = x - y;
        //if (nuw && y && x > std::numeric_limits<UT>::max() / y) return {};
        // TODO nsw
        return res;
    }
};

template<nat_t w> struct Fold<WOp, WOp::mul, w> {
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

template<nat_t w> struct Fold<WOp, WOp::shl, w> {
    static Res run(u64 a, u64 b, bool nsw, bool nuw) {
        using T = w2u<w>;
        auto x = get<T>(a), y = get<T>(b);
        if (u64(y) > w) return {};
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

template<nat_t w> struct Fold<ZOp, ZOp::sdiv, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } };
template<nat_t w> struct Fold<ZOp, ZOp::udiv, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } };
template<nat_t w> struct Fold<ZOp, ZOp::smod, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } };
template<nat_t w> struct Fold<ZOp, ZOp::umod, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } };

template<nat_t w> struct Fold<Shr, Shr::   a, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } };
template<nat_t w> struct Fold<Shr, Shr::   l, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } };

template<nat_t w> struct Fold<ROp, ROp:: add, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) + get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: sub, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) - get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: mul, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) * get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: div, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(get<T>(a) / get<T>(b)); } };
template<nat_t w> struct Fold<ROp, ROp:: mod, w> { static Res run(u64 a, u64 b) { using T = w2r<w>; return T(rem(get<T>(a), get<T>(b))); } };

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

/// @attention Note that @p a and @p b are passed by reference as fold also commutes if possible. See commute().
template<class Op, Op op>
static const Def* fold(World& world, const Def* type, const App* callee, const Def*& a, const Def*& b, const Def* dbg) {
    static constexpr int min_w = std::is_same_v<Op, ROp> || std::is_same_v<Op, RCmp> ? 16 : 1;
    auto la = a->isa<Lit>(), lb = b->isa<Lit>();

    if (a->isa<Bot>() || b->isa<Bot>()) return world.bot(type, dbg);

    if (la && lb) {
        nat_t width;
        [[maybe_unused]] bool nsw = false, nuw = false;
        if constexpr (std::is_same_v<Op, WOp>) {
            auto [mode, w] = callee->args<2>(as_lit<nat_t>);
            nsw = mode & WMode::nsw;
            nuw = mode & WMode::nuw;
            width = w;
        } else {
            width = as_lit<nat_t>(a->type()->as<App>()->arg());
        }

        Res res;
        switch (width) {
#define CODE(i)                                                                     \
            case i:                                                                 \
                if constexpr (i >= min_w) {                                         \
                    if constexpr (std::is_same_v<Op, WOp>)                          \
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

    if (is_commutative(op)) {
        if (lb || (a->gid() > b->gid() && !la)) // swap lit to left, or smaller gid to left if no lit present
            std::swap(a, b);
    }

    return nullptr;
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
            auto app_m = isa_lit<nat_t>(app->arg(0));
            if (!app_m || !has(*app_m, RMode::reassoc)) return false;
            rmode &= *app_m; // least upper bound
            return true;
        };

        if (!check_mode(ab)) return nullptr;
        if (lx && !check_mode(xy->decurry())) return nullptr;
        if (lz && !check_mode(zw->decurry())) return nullptr;

        make_op = [&](const Def* a, const Def* b) { return world.op(op, rmode, a, b, dbg); };
    } else if constexpr (tag == Tag::WOp) {
        // if we reassociate WOps, we have to forget about nsw/nuw
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

/*
 * normalize
 */

template<Shr op>
const Def* normalize_Shr(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->split<2>();
    auto w = isa_lit<nat_t>(callee->arg());

    if (auto result = fold<Shr, op>(world, type, callee, a, b, dbg)) return result;

    if (auto la = a->isa<Lit>()) {
        if (la == world.lit_int(*w, 0)) {
            switch (op) {
                case Shr::a: return la;
                case Shr::l: return la;
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (auto lb = b->isa<Lit>()) {
        if (lb == world.lit_int(*w, 0)) {
            switch (op) {
                case Shr::a: return a;
                case Shr::l: return a;
                default: THORIN_UNREACHABLE;
            }
        }

        if (lb->get() > *w) return world.bot(type, dbg);
    }

    return world.raw_app(callee, {a, b}, dbg);
}

template<WOp op>
const Def* normalize_WOp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->split<2>();
    auto [m, w] = callee->args<2>(isa_lit<nat_t>); // mode and width

    if (auto result = fold<WOp, op>(world, type, callee, a, b, dbg)) return result;

    if (auto la = a->isa<Lit>()) {
        if (la == world.lit_int(*w, 0)) {
            switch (op) {
                case WOp::add: return b;    // 0  + b -> b
                case WOp::sub: break;
                case WOp::mul: return la;   // 0  * b -> 0
                case WOp::shl: return la;   // 0 << b -> 0
                default: THORIN_UNREACHABLE;
            }
        }

        if (la == world.lit_int(*w, 1)) {
            switch (op) {
                case WOp::add: break;
                case WOp::sub: break;
                case WOp::mul: return b;    // 1  * b -> b
                case WOp::shl: break;
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (auto lb = b->isa<Lit>()) {
        if (lb == world.lit_int(*w, 0)) {
            switch (op) {
                case WOp::sub: return a;    // a  - 0 -> a
                case WOp::shl: return a;    // a >> 0 -> a
                default: THORIN_UNREACHABLE;
                // add, mul are commutative, the literal has been normalized to the left
            }
        }

        if (op == WOp::sub)
            return world.op(WOp::add, *m, a, world.lit_int(*w, ~lb->get() + 1_u64)); // a - lb -> a + (~lb + 1)
        else if (op == WOp::shl && lb->get() > *w)
            return world.bot(type, dbg);
    }

    if (a == b) {
        switch (op) {
            case WOp::add: return world.op(WOp::mul, *m, world.lit_int(*w, 2), a, dbg); // a + a -> 2 * a
            case WOp::sub: return world.lit_int(*w, 0);                                 // a - a -> 0
            case WOp::mul: break;
            case WOp::shl: break;
            default: THORIN_UNREACHABLE;
        }
    }

    if (auto res = reassociate<Tag::WOp>(op, world, callee, a, b, dbg)) return res;

    return world.raw_app(callee, {a, b}, dbg);
}

template<ZOp op>
const Def* normalize_ZOp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [mem, a, b] = arg->split<3>();
    auto w = isa_lit<nat_t>(callee->arg());
    type = type->as<Sigma>()->op(1); // peel of actual type
    auto make_res = [&](const Def* res) { return world.tuple({mem, res}, dbg); };

    if (auto result = fold<ZOp, op>(world, type, callee, a, b, dbg)) return make_res(result);

    if (auto la = a->isa<Lit>()) {
        if (la == world.lit_int(*w, 0)) return make_res(la); // 0 / b -> 0 and 0 % b -> 0
    }

    if (auto lb = b->isa<Lit>()) {
        if (lb == world.lit_int(*w, 0)) return make_res(world.bot(type)); // a / 0 -> ⊥ and a % 0 -> ⊥

        if (lb == world.lit_int(*w, 1)) {
            switch (op) {
                case ZOp::sdiv: return make_res(a);                    // a / 1 -> a
                case ZOp::udiv: return make_res(a);                    // a / 1 -> a
                case ZOp::smod: return make_res(world.lit_int(*w, 0)); // a % 1 -> 0
                case ZOp::umod: return make_res(world.lit_int(*w, 0)); // a % 1 -> 0
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (a == b) {
        switch (op) {
            case ZOp::sdiv: return make_res(world.lit_int(*w, 1)); // a / a -> 1
            case ZOp::udiv: return make_res(world.lit_int(*w, 1)); // a / a -> 1
            case ZOp::smod: return make_res(world.lit_int(*w, 0)); // a % a -> 0
            case ZOp::umod: return make_res(world.lit_int(*w, 0)); // a % a -> 0
            default: THORIN_UNREACHABLE;
        }
    }

    return world.raw_app(callee, {mem, a, b}, dbg);
}

template<ROp op>
const Def* normalize_ROp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->split<2>();
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
                    case ROp::mod: return la;   // 0 % b -> 0
                    default: THORIN_UNREACHABLE;
                }
            }

            if (la == world.lit_real(*w, 1.0)) {
                switch (op) {
                    case ROp::add: break;
                    case ROp::sub: break;
                    case ROp::mul: return b;    // 1  * b -> b
                    case ROp::div: break;
                    case ROp::mod: break;
                    default: THORIN_UNREACHABLE;
                }
            }
        }

        if (auto lb = b->isa<Lit>()) {
            if (lb == world.lit_real(*w, 0.0)) {
                switch (op) {
                    case ROp::sub: return a;    // a - 0 -> a
                    case ROp::div: break;
                    case ROp::mod: break;
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
                case ROp::mod: break;
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
    auto [a, b] = arg->split<2>();

    if (auto result = fold<ICmp, op>(world, type, callee, a, b, dbg)) return result;
    if constexpr (op == ICmp::_f) return world.lit_false();
    if constexpr (op == ICmp::_t) return world.lit_true();

    return world.raw_app(callee, {a, b}, dbg);
}

template<RCmp op>
const Def* normalize_RCmp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->split<2>();

    if (auto result = fold<RCmp, op>(world, type, callee, a, b, dbg)) return result;
    if constexpr (op == RCmp::f) return world.lit_false();
    if constexpr (op == RCmp::t) return world.lit_true();

    return world.raw_app(callee, {a, b}, dbg);
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
        if (arg->is_const()) return world.lit_true();
    } else {
        if (world.is_pe_done()) return arg;
    }

    return world.raw_app(callee, arg, dbg);
}

const Def* normalize_bit(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [tbl, a, b] = arg->split<3>();
    auto w = isa_lit<nat_t>(callee->arg());

    if (!tbl->is_const() || !w) return world.raw_app(callee, arg, dbg);

    if (tbl == world.table(Bit::    f)) return world.lit_int(*w,      0);
    if (tbl == world.table(Bit::    t)) return world.lit_int(*w, u64(-1));
    if (tbl == world.table(Bit::    a)) return a;
    if (tbl == world.table(Bit::    b)) return b;
    if (tbl == world.table(Bit::   na)) return world.op_bit_not(a, dbg);
    if (tbl == world.table(Bit::   nb)) return world.op_bit_not(b, dbg);
    if (tbl == world.table(Bit:: ciff)) return world.op(Bit:: iff, b, a, dbg);
    if (tbl == world.table(Bit::nciff)) return world.op(Bit::niff, b, a, dbg);

    if (a->isa<Lit>() && b->isa<Lit>()) {
        u64 x = a->as<Lit>()->get();
        u64 y = b->as<Lit>()->get();
        u64 res;

        if (false) {}
        else if (tbl == world.table(Bit::_and)) res =   x & y;
        else if (tbl == world.table(Bit:: _or)) res =   x | y;
        else if (tbl == world.table(Bit::_xor)) res =   x ^ y;
        else if (tbl == world.table(Bit::nand)) res = ~(x & y);
        else if (tbl == world.table(Bit:: nor)) res = ~(x | y);
        else if (tbl == world.table(Bit::nxor)) res = ~(x ^ y);
        else if (tbl == world.table(Bit:: iff)) res = ~x |  y;
        else if (tbl == world.table(Bit::niff)) res =  x & ~y;
        else THORIN_UNREACHABLE;

        return world.lit_int(*w, res);
    }

    bool sym = is_symmetric(tbl);

    if (sym) {
        if (b->isa<Lit>()) std::swap(a, b); // commute literals to a

        auto la = a->isa<Lit>();
        auto xy = isa<Tag::Bit>(a);
        auto zw = isa<Tag::Bit>(b);
        if (xy && xy->arg(0) != tbl) xy.clear();
        if (zw && zw->arg(0) != tbl) zw.clear();
        auto lx = xy ? xy->arg(1)->template isa<Lit>() : nullptr;
        auto lz = zw ? zw->arg(1)->template isa<Lit>() : nullptr;
        auto  y = xy ? xy->arg(2) : nullptr;
        auto  w = zw ? zw->arg(2) : nullptr;

        if (la && lz) return world.op_bit(tbl, world.op_bit(tbl, la, lz), w);                       // (1)
        if (lx && lz) return world.op_bit(tbl, world.op_bit(tbl, lx, lz), world.op_bit(tbl, y, w)); // (2)
        if (lz)       return world.op_bit(tbl, lz, world.op_bit(tbl, a, w));                        // (3)
        if (lx)       return world.op_bit(tbl, lx, world.op_bit(tbl, y, b));                        // (4)
    }

    auto make_lit = [&](const Def* def) {
        return as_lit<bool>(def) ? world.lit_int(*w, u64(-1)) : world.lit_int(*w, 0);
    };

    if (a == world.lit_int(*w, 0) || a == world.lit_int(*w, u64(-1))) {
        auto row = proj(tbl, 2, as_lit<u64>(a) ? 1 : 0);
        if (auto pack = row->isa<Pack>()) return make_lit(pack->body());
        if (row == world.table_id())      return b;
        if (row == world.table_not() && tbl != world.table(Bit::_xor)) return world.op_bit_not(b, dbg);
    }

    if (sym && a == b) {
        auto x = as_lit<bool>(proj(proj(tbl, 2, 0), 2, 0));
        auto y = as_lit<bool>(proj(proj(tbl, 2, 1), 2, 1));
        if (!x && !y) return world.lit_int(*w, 0);
        if ( x &&  y) return world.lit_int(*w, 1);
        if (!x &&  y) return a;
        if ( x && !y) return world.op_bit_not(a, dbg);
        THORIN_UNREACHABLE;
    }

    return world.raw_app(callee, {tbl, a, b}, dbg);
}

#if 0
template<IOp op>
const Def* normalize_IOp(const Def* type, const Def* c, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto callee = c->as<App>();
    auto [a, b] = arg->split<2>();
    auto w = isa_lit<nat_t>(callee->arg());

    if (auto result = fold<IOp, op>(world, type, callee, a, b, dbg)) return result;

    if (auto la = a->isa<Lit>()) {
        if (la == world.lit_int(*w, 0)) {
            switch (op) {
                case IOp::iand: return la;
                case IOp::ior : return b;
                case IOp::ixor: return b;
                default: THORIN_UNREACHABLE;
            }
        }

        if (la == world.lit_int(*w, u64(-1))) {
            switch (op) {
                case IOp::iand: return b;
                case IOp::ior : return la;
                case IOp::ixor: break;
                default: THORIN_UNREACHABLE;
            }
        }
    }

    if (a == b) {
        switch (op) {
            case IOp::iand: return a;
            case IOp::ior : return a;
            case IOp::ixor: return world.lit_int(*w, 0);
            default: THORIN_UNREACHABLE;
        }
    }

    // commute NOT to b
    if (is_commutative(op) && is_not(a)) std::swap(a, b);

    if (auto bb = is_not(b); bb && a == bb) {
        switch (op) {
            case IOp::iand: return world.lit_int(*w,       0);
            case IOp::ior : return world.lit_int(*w, u64(-1));
            case IOp::ixor: return world.lit_int(*w, u64(-1));
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
    if (auto res = reassociate<Tag::IOp>(op, world, callee, a, b, dbg)) return res;

    return world.raw_app(callee, {a, b}, dbg);
}
#endif

const Def* normalize_bitcast(const Def* dst_type, const Def* callee, const Def* src, const Def* dbg) {
    auto& world = dst_type->world();

    if (src->isa<Bot>())         return world.bot(dst_type);
    if (src->type() == dst_type) return src;

    if (auto other = isa<Tag::Bitcast>(src))
        return other->arg()->type() == dst_type ? other->arg() : world.op_bitcast(dst_type, other->arg(), dbg);

    if (auto lit = src->isa<Lit>()) {
        if (dst_type->isa<Nat>()) return world.lit(dst_type, lit->get(), dbg);
        if (get_width(dst_type))  return world.lit(dst_type, lit->get(), dbg);

        if (auto a = isa_lit_arity(dst_type)) {
            if (lit->get() < *a) return world.lit_index(dst_type, lit->get(), dbg);
            return world.bot(dst_type, dbg); // this was an unsound cast - so return bottom
        }
    }

    return world.raw_app(callee, src, dbg);
}

const Def* normalize_lea(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [ptr, index] = arg->split<2>();
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();

    if (isa_lit_arity(pointee->arity(), 1)) return ptr;

    return world.raw_app(callee, {ptr, index}, dbg);
}

const Def* normalize_sizeof(const Def* t, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = t->world();

    if (auto w = get_width(arg)) return world.lit_nat(*w / 8, dbg);

    return world.raw_app(callee, arg, dbg);
}

const Def* normalize_load(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [mem, ptr] = arg->split<2>();
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();

    if (ptr->isa<Bot>()) return world.tuple({mem, world.bot(type->as<Sigma>()->op(1))}, dbg);

    // loading an empty tuple can only result in an empty tuple
    if (auto sigma = pointee->isa<Sigma>(); sigma && sigma->num_ops() == 0)
        return world.tuple({mem, world.tuple(sigma->type(), {}, dbg)});

    return world.raw_app(callee, {mem, ptr}, dbg);
}

const Def* normalize_store(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [mem, ptr, val] = arg->split<3>();

    if (ptr->isa<Bot>() || val->isa<Bot>()) return mem;
    if (auto pack = val->isa<Pack>(); pack && pack->body()->isa<Bot>()) return mem;
    if (auto tuple = val->isa<Tuple>()) {
        if (std::all_of(tuple->ops().begin(), tuple->ops().end(), [&](const Def* op) { return op->isa<Bot>(); }))
            return mem;
    }

    return world.raw_app(callee, {mem, ptr, val}, dbg);
}

/*
 * instantiate templates
 */

#define CODE(T, o) template const Def* normalize_ ## T<T::o>(const Def*, const Def*, const Def*, const Def*);
THORIN_SHR  (CODE)
THORIN_W_OP (CODE)
THORIN_Z_OP (CODE)
THORIN_R_OP (CODE)
THORIN_I_CMP(CODE)
THORIN_R_CMP(CODE)
THORIN_CONV (CODE)
THORIN_PE   (CODE)
#undef CODE

}
