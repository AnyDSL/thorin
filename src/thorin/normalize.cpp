#include "thorin/def.h"
#include "thorin/fold.h"
#include "thorin/world.h"

namespace thorin {

using namespace thorin::fold;

static std::array<const Def*, 2> split(const Def* def) {
    auto& w = def->world();
    auto a = w.extract(def, 0_u64);
    auto b = w.extract(def, 1_u64);
    return {{a, b}};
}

const Def* normalize_select(const Def* callee, const Def* arg, const Def* dbg) {
    auto& world = callee->world();

    auto cond = world.extract(arg, 0_s);
    auto a = world.extract(arg, 1);
    auto b = world.extract(arg, 2);

    if (cond->isa<Bot>() || a->isa<Bot>() || b->isa<Bot>()) return world.bot(a->type(), dbg);
    if (auto lit = cond->isa<Lit>()) return lit->get<bool>() ? a : b;

#if 0
    if (is_not(cond)) {
        cond = cond->as<ArithOp>()->rhs();
        std::swap(a, b);
    }
#endif

    if (a == b) return a;
    return nullptr;
}

//const Def* normalize_sizeof(const Def* callee, const Def* arg, const Def* dbg) {
    //auto& world = callee->world();
    //if (auto uint = type->isa<Uint>()) return lit_nat(uint->lit_num_bits() / 8_u64, false, dbg);
    //if (auto real = type->isa<Real>()) return lit_nat(real->lit_num_bits() / 8_u64, false, dbg);
    //return 0;
//}

template<template<int, bool, bool> class F>
static const Def* try_wfold(const Def* callee, const Def* a, const Def* b, const Def* dbg) {
    auto& world = callee->world();
    auto la = a->isa<Lit>(), lb = b->isa<Lit>();
    if (la && lb) {
        auto t = a->type();
        auto fw = callee->as<App>()->arg();
        auto f = as_lit<u64>(world.extract(fw, 0_u64));
        auto w = as_lit<u64>(world.extract(fw, 1_u64));
        Res res;
        switch (f) {
            case u64(WMode::none):
                switch (w) {
                    case  8: res = F< 8, false, false>::run(la->get(), lb->get()); break;
                    case 16: res = F<16, false, false>::run(la->get(), lb->get()); break;
                    case 32: res = F<32, false, false>::run(la->get(), lb->get()); break;
                    case 64: res = F<64, false, false>::run(la->get(), lb->get()); break;
                    default: THORIN_UNREACHABLE;
                }
                break;
            case u64(WMode::nsw):
                switch (w) {
                    case  8: res = F< 8,  true, false>::run(la->get(), lb->get()); break;
                    case 16: res = F<16,  true, false>::run(la->get(), lb->get()); break;
                    case 32: res = F<32,  true, false>::run(la->get(), lb->get()); break;
                    case 64: res = F<64,  true, false>::run(la->get(), lb->get()); break;
                    default: THORIN_UNREACHABLE;
                }
                break;
            case u64(WMode::nuw):
                switch (w) {
                    case  8: res = F< 8, false,  true>::run(la->get(), lb->get()); break;
                    case 16: res = F<16, false,  true>::run(la->get(), lb->get()); break;
                    case 32: res = F<32, false,  true>::run(la->get(), lb->get()); break;
                    case 64: res = F<64, false,  true>::run(la->get(), lb->get()); break;
                    default: THORIN_UNREACHABLE;
                }
                break;
            case u64(WMode::nsw | WMode::nuw):
                switch (w) {
                    case  8: res = F< 8,  true,  true>::run(la->get(), lb->get()); break;
                    case 16: res = F<16,  true,  true>::run(la->get(), lb->get()); break;
                    case 32: res = F<32,  true,  true>::run(la->get(), lb->get()); break;
                    case 64: res = F<64,  true,  true>::run(la->get(), lb->get()); break;
                    default: THORIN_UNREACHABLE;
                }
                break;
        }

        if (res) return world.lit(t, *res, dbg);
        return world.bot(t, dbg);
    }

    return nullptr;
}

template<WOp op>
const Def* normalize_WOp(const Def* callee, const Def* arg, const Def* dbg) {
    auto [a, b] = split(arg);
    if (auto result = try_wfold<FoldWOp<op>::template Fold>(callee, a, b, dbg)) return result;

    return nullptr;
}

template<ZOp op>
const Def* normalize_ZOp(const Def*, const Def*, const Def*) { return nullptr; }

template<template<int> class F>
static const Def* try_ifold(const Def* callee, const Def* a, const Def* b, const Def* dbg) {
    auto& world = callee->world();
    auto la = a->isa<Lit>(), lb = b->isa<Lit>();
    if (la && lb) {
        auto t = a->type();
        auto w = as_lit<u64>(t->as<App>()->arg());
        Res res;
        switch (w) {
            case  1: res = F< 1>::run(la->get(), lb->get()); break;
            case  8: res = F< 8>::run(la->get(), lb->get()); break;
            case 16: res = F<16>::run(la->get(), lb->get()); break;
            case 32: res = F<32>::run(la->get(), lb->get()); break;
            case 64: res = F<64>::run(la->get(), lb->get()); break;
            default: THORIN_UNREACHABLE;
        }

        if (res) return world.lit(t, *res, dbg);
        return world.bot(t, dbg);
    }

    return nullptr;
}

template<IOp op>
const Def* normalize_IOp(const Def* callee, const Def* arg, const Def* dbg) {
    auto [a, b] = split(arg);
    if (auto result = try_ifold<FoldIOp<op>::template Fold>(callee, a, b, dbg)) return result;

    return nullptr;
}

template<ROp op>
const Def* normalize_ROp(const Def*, const Def*, const Def*) { return nullptr; }

template<ICmp op>
const Def* normalize_ICmp(const Def*, const Def*, const Def*) { return nullptr; }

template<RCmp op>
const Def* normalize_RCmp(const Def*, const Def*, const Def*) { return nullptr; }

template<Cast op>
const Def* normalize_Cast(const Def*, const Def*, const Def*) { return nullptr; }

// instantiate templates
#define CODE(T, o) template const Def* normalize_ ## T<T::o>(const Def*, const Def*, const Def*);
    THORIN_W_OP (CODE)
    THORIN_Z_OP (CODE)
    THORIN_I_OP (CODE)
    THORIN_R_OP (CODE)
    THORIN_I_CMP(CODE)
    THORIN_R_CMP(CODE)
    THORIN_CAST(CODE)
#undef CODE

}
