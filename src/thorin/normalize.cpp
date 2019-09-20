#include "thorin/def.h"
#include "thorin/util.h"
#include "thorin/world.h"
#include "thorin/util/log.h"

namespace thorin {

/*
 * helpers
 */

static bool is_allset(const Def* def) {
    if (auto lit = isa_lit<u64>(def)) {
        if (auto w = get_width(def->type()))
            return def == def->world().lit_int_max(*w);
    }
    return false;
}

static const Def* is_not(const Def* def) {
    if (auto ixor = isa<Tag::IOp>(IOp::ixor, def)) {
        auto [x, y] = ixor->args<2>();
        if (is_allset(x)) return y;
    }
    return nullptr;
}

template<class T> static T get(u64 u) { return bitcast<T>(u); }

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
    static Res run(u64 a, u64 b, bool /*nsw*/, bool nuw) {
        using UT = w2u<w>;
        auto x = get<UT>(a), y = get<UT>(b);
        decltype(x) res = x * y;
        if (nuw && y && x > std::numeric_limits<UT>::max() / y) return {};
        // TODO nsw
        return res;
    }
};

template<nat_t w> struct Fold<WOp, WOp::shl, w> {
    static Res run(u64 a, u64 b, bool nsw, bool nuw) {
        using T = w2u<w>;
        auto x = get<T>(a), y = get<T>(b);
        if (y > w) return {};
        decltype(x) res = x << y;
        if (nuw && res < x) return {};
        if (nsw && get_sign(x) != get_sign(res)) return {};
        return res;
    }
};

template<nat_t w> struct Fold<ZOp, ZOp::sdiv, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } };
template<nat_t w> struct Fold<ZOp, ZOp::udiv, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) / r); } };
template<nat_t w> struct Fold<ZOp, ZOp::smod, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } };
template<nat_t w> struct Fold<ZOp, ZOp::umod, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; T r = get<T>(b); if (r == 0) return {}; return T(get<T>(a) % r); } };

template<nat_t w> struct Fold<IOp, IOp::ashr, w> { static Res run(u64 a, u64 b) { using T = w2s<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } };
template<nat_t w> struct Fold<IOp, IOp::lshr, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; if (b > w) return {}; return T(get<T>(a) >> get<T>(b)); } };
template<nat_t w> struct Fold<IOp, IOp::iand, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) & get<T>(b)); } };
template<nat_t w> struct Fold<IOp, IOp::ior , w> { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) | get<T>(b)); } };
template<nat_t w> struct Fold<IOp, IOp::ixor, w> { static Res run(u64 a, u64 b) { using T = w2u<w>; return T(get<T>(a) ^ get<T>(b)); } };

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
template<nat_t sw, nat_t dw> struct FoldConv<Conv::s2s, sw, dw> { static Res run(u64 src) { return w2s<dw>(get<w2s<sw>>(src)); } };
template<nat_t sw, nat_t dw> struct FoldConv<Conv::u2u, sw, dw> { static Res run(u64 src) { return w2u<dw>(get<w2u<sw>>(src)); } };
template<nat_t sw, nat_t dw> struct FoldConv<Conv::s2r, sw, dw> { static Res run(u64 src) { return w2r<dw>(get<w2s<sw>>(src)); } };
template<nat_t sw, nat_t dw> struct FoldConv<Conv::u2r, sw, dw> { static Res run(u64 src) { return w2r<dw>(get<w2u<sw>>(src)); } };
template<nat_t sw, nat_t dw> struct FoldConv<Conv::r2s, sw, dw> { static Res run(u64 src) { return w2s<dw>(get<w2r<sw>>(src)); } };
template<nat_t sw, nat_t dw> struct FoldConv<Conv::r2u, sw, dw> { static Res run(u64 src) { return w2u<dw>(get<w2r<sw>>(src)); } };
template<nat_t sw, nat_t dw> struct FoldConv<Conv::r2r, sw, dw> { static Res run(u64 src) { return w2r<dw>(get<w2r<sw>>(src)); } };

/*
 * fold
 */

template<nat_t min_w, class Op, Op op>
static const Def* fold(const Def* type, const Def* callee, const Def* m, const Def* a, const Def* b, const Def* dbg) {
    auto& world = type->world();
    if (m) type = type->as<Sigma>()->op(1); // peel of actual type for ZOps

    if (a->isa<Bot>() || b->isa<Bot>() || (m != nullptr && m->isa<Bot>())) {
        auto bot = world.bot(type, dbg);
        return m ? world.tuple({m, bot}) : bot;
    }

    [[maybe_unused]] bool nsw = false, nuw = false;
    if constexpr (std::is_same<Op, WOp>()) {
        auto [f, w] = callee->as<App>()->args<2>(isa_lit<nat_t>);
        if (!f && !w) return nullptr;
        nsw = *f & WMode::nsw;
        nuw = *f & WMode::nuw;
    }

    auto la = a->isa<Lit>(), lb = b->isa<Lit>();
    if (la && lb) {
        auto w = as_lit<nat_t>(a->type()->as<App>()->arg());
        Res res;
        switch (w) {
#define CODE(i)                                                                     \
            case i:                                                                 \
                if constexpr (i >= min_w) {                                         \
                    if constexpr (std::is_same<Op, WOp>())                          \
                        res = Fold<Op, op, i>::run(la->get(), lb->get(), nsw, nuw); \
                    else                                                            \
                        res = Fold<Op, op, i>::run(la->get(), lb->get());           \
                }                                                                   \
                break;
            THORIN_1_8_16_32_64(CODE)
#undef CODE
            default: THORIN_UNREACHABLE;
        }

        auto result = res ? world.lit(type, *res, dbg) : world.bot(type, dbg);
        return m ? world.tuple({m, result}, dbg) : result;
    }

    return nullptr;
}

#define TABLE(m) m( 1,  1) m( 1,  8) m( 1, 16) m( 1, 32) m( 1, 64) \
                 m( 8,  1) m( 8,  8) m( 8, 16) m( 8, 32) m( 8, 64) \
                 m(16,  1) m(16,  8) m(16, 16) m(16, 32) m(16, 64) \
                 m(32,  1) m(32,  8) m(32, 16) m(32, 32) m(32, 64) \
                 m(64,  1) m(64,  8) m(64, 16) m(64, 32) m(64, 64)

template<nat_t min_sw, nat_t min_dw, Conv op>
static const Def* fold_Conv(const Def* dst_type, const Def* callee, const Def* src, const Def* dbg) {
    auto& world = dst_type->world();
    if (src->isa<Bot>()) return world.bot(dst_type, dbg);

    auto [lit_sw, lit_dw] = callee->as<App>()->args<2>(isa_lit<nat_t>);
    auto lit_src = src->isa<Lit>();
    if (lit_src && lit_sw && lit_dw) {
        Res res;
#define CODE(sw, dw)                                             \
        else if (*lit_sw == sw && *lit_dw == dw) {               \
            if constexpr (sw >= min_sw && dw >= min_dw)          \
                res = FoldConv<op, sw, dw>::run(lit_src->get()); \
        }
        if (false) {} TABLE(CODE)
#undef CODE
        if (res) return world.lit(dst_type, *res, dbg);
        return world.bot(dst_type, dbg);
    }

    return nullptr;
}

/*
 * normalize
 */

template<tag_t tag, class F>
static const Def* merge_cmps(World& world, const Def* a, const Def* b) {
    auto a_cmp = isa<tag>(a), b_cmp = isa<tag>(b);
    if (a_cmp && b_cmp && a_cmp->arg() == b_cmp->arg()) {
        auto [x, y] = a_cmp->template args<2>();
        return world.op(Tag2Enum<tag>(F()(flags_t(a_cmp.flags()), flags_t(b_cmp.flags()))), x, y);
    }
    return nullptr;
}

template<class F>
static const Def* merge_cmps(World& world, const Def* a, const Def* b) {
    if (auto res = merge_cmps<Tag::ICmp, F>(world, a, b)) return res;
    if (auto res = merge_cmps<Tag::RCmp, F>(world, a, b)) return res;
    return nullptr;
}

template<IOp op>
const Def* normalize_IOp(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [a, b] = arg->split<2>();
    if (auto result = fold<1, IOp, op>(type, callee, nullptr, a, b, dbg)) return result;

    if (op == IOp::ixor) {
        if (is_allset(a)) { // bitwise not
            if (auto icmp = isa<Tag::ICmp>(b)) { auto [x, y] = icmp->args<2>(); return world.op(ICmp(~flags_t(icmp.flags()) & 0b11111), y, x); }
            if (auto rcmp = isa<Tag::RCmp>(b)) { auto [x, y] = rcmp->args<2>(); return world.op(RCmp(~flags_t(rcmp.flags()) & 0b01111), y, x); }
        }
        if (auto res = merge_cmps<std::bit_xor<flags_t>>(world, a, b)) return res;
    } else if (op == IOp::iand) {
        if (auto res = merge_cmps<std::bit_and<flags_t>>(world, a, b)) return res;
    } else if (op == IOp::ior) {
        if (auto res = merge_cmps<std::bit_or <flags_t>>(world, a, b)) return res;
    }

    return nullptr;
}

template<WOp op>
const Def* normalize_WOp(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto [a, b] = arg->split<2>();
    if (auto result = fold<8, WOp, op>(type, callee, nullptr, a, b, dbg)) return result;

    return nullptr;
}

template<ZOp op>
const Def* normalize_ZOp(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto [m, a, b] = arg->split<3>();
    if (auto result = fold<8, ZOp, op>(type, callee, m, a, b, dbg)) return result;

    return nullptr;
}

template<ROp op>
const Def* normalize_ROp(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto [a, b] = arg->split<2>();
    if (auto result = fold<16, ROp, op>(type, callee, nullptr, a, b, dbg)) return result;

    return nullptr;
}

template<ICmp op>
const Def* normalize_ICmp(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [a, b] = arg->split<2>();

    if (auto result = fold<1, ICmp, op>(type, callee, nullptr, a, b, dbg)) return result;
    if constexpr (op == ICmp::_f) return world.lit_false();
    if constexpr (op == ICmp::_t) return world.lit_true();

    return nullptr;
}

template<RCmp op>
const Def* normalize_RCmp(const Def* type, const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = type->world();

    auto [a, b] = arg->split<2>();
    if (auto result = fold<16, RCmp, op>(type, callee, nullptr, a, b, dbg)) return result;
    if constexpr (op == RCmp::f) return world.lit_false();
    if constexpr (op == RCmp::t) return world.lit_true();

    return nullptr;
}

template<Conv op>
const Def* normalize_Conv(const Def* dst_type, const Def* callee, const Def* src, const Def* dbg) {
    auto& world = dst_type->world();

    static constexpr auto min_sw = op == Conv::r2s || op == Conv::r2u || op == Conv::r2r ? 16 : 1;
    static constexpr auto min_dw = op == Conv::s2r || op == Conv::u2r || op == Conv::r2r ? 16 : 1;
    if (auto result = fold_Conv<min_sw, min_dw, op>(dst_type, callee, src, dbg)) return result;

    auto [sw, dw] = callee->as<App>()->args<2>(isa_lit<nat_t>);
    if (sw == dw && dst_type == src->type()) return src;

    if constexpr (op == Conv::s2s) {
        if (sw && dw && *sw < *dw) return world.op(Conv::u2u, dst_type, src, dbg);
    }

    return nullptr;
}

template<PE op>
const Def* normalize_PE(const Def* type, const Def*, const Def* arg, const Def*) {
    auto& world = type->world();

    if constexpr (op == PE::known) {
        if (world.is_pe_done() || isa<Tag::PE>(PE::hlt, arg)) return world.lit_false();
        if (is_const(arg)) return world.lit_true();
    } else {
        if (world.is_pe_done()) return arg;
    }

    return nullptr;
}

const Def* normalize_bitcast(const Def* dst_type, const Def*, const Def* src, const Def* dbg) {
    auto& world = dst_type->world();

    if (src->isa<Bot>())                     return world.bot(dst_type);
    if (src->type() == dst_type)             return src;
    if (auto other = isa<Tag::Bitcast>(src)) return world.op_bitcast(dst_type, other->arg(), dbg);

    if (auto lit = src->isa<Lit>()) {
        if (is_arity(dst_type))           return world.lit_index(dst_type, lit->get());
        if (dst_type->isa<Nat>())         return world.lit(dst_type, lit->get());
        if (auto w = get_width(dst_type)) return world.lit(dst_type, (u64(-1) >> (64_u64 - *w)) & lit->get());
    }

    if (auto variant = src->isa<Variant>()) {
        if (variant->op(0)->type() != dst_type) ELOG("variant downcast not possible");
        return variant->op(0);
    }

    return nullptr;
}

const Def* normalize_lea(const Def* type, const Def*, const Def* arg, const Def*) {
    auto& world = type->world();
    auto [ptr, index] = arg->split<2>();
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();

    if (pointee->arity() == world.lit_arity_1()) return ptr;

    return nullptr;
}

const Def* normalize_select(const Def* type, const Def*, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [cond, a, b] = arg->split<3>();

    if (cond->isa<Bot>())            return world.bot(type, dbg);
    if (a == b)                      return a;
    if (auto lit = cond->isa<Lit>()) return lit->get<bool>() ? a : b;
    if (auto neg = is_not(cond))     return world.op_select(neg, b, a, dbg);

    return nullptr;
}

const Def* normalize_sizeof(const Def*, const Def*, const Def* type, const Def* dbg) {
    auto& world = type->world();

    if (auto w = get_width(type)) return world.lit_nat(*w / 8, dbg);
    return nullptr;
}

const Def* normalize_load(const Def* type, const Def*, const Def* arg, const Def* dbg) {
    auto& world = type->world();
    auto [mem, ptr] = arg->split<2>();
    auto [pointee, addr_space] = as<Tag::Ptr>(ptr->type())->args<2>();

    if (ptr->isa<Bot>()) return world.tuple({mem, world.bot(type->as<Sigma>()->op(1))}, dbg);

    // loading an empty tuple can only result in an empty tuple
    if (auto sigma = pointee->isa<Sigma>(); sigma && sigma->num_ops() == 0)
        return world.tuple({mem, world.tuple(sigma->type(), {}, dbg)});

    return nullptr;
}

const Def* normalize_store(const Def*, const Def*, const Def* arg, const Def*) {
    auto [mem, ptr, val] = arg->split<3>();

    if (ptr->isa<Bot>() || val->isa<Bot>()) return mem;
    if (auto pack = val->isa<Pack>(); pack && pack->body()->isa<Bot>()) return mem;
    if (auto tuple = val->isa<Tuple>()) {
        if (std::all_of(tuple->ops().begin(), tuple->ops().end(), [&](const Def* op) { return op->isa<Bot>(); }))
            return mem;
    }

    return nullptr;
}

/*
 * instantiate templates
 */

#define CODE(T, o) template const Def* normalize_ ## T<T::o>(const Def*, const Def*, const Def*, const Def*);
THORIN_W_OP (CODE)
THORIN_Z_OP (CODE)
THORIN_I_OP (CODE)
THORIN_R_OP (CODE)
THORIN_I_CMP(CODE)
THORIN_R_CMP(CODE)
THORIN_CONV (CODE)
THORIN_PE   (CODE)
#undef CODE

}
