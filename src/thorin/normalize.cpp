#include "thorin/def.h"
#include "thorin/world.h"

namespace thorin {

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

template<WOp op>
const Def* normalize_WOp(const Def*, const Def*, const Def*) { return nullptr; }

template<ZOp op>
const Def* normalize_ZOp(const Def*, const Def*, const Def*) { return nullptr; }

template<IOp op>
const Def* normalize_IOp(const Def*, const Def*, const Def*) { return nullptr; }

template<ROp op>
const Def* normalize_ROp(const Def*, const Def*, const Def*) { return nullptr; }

template<ICmp op>
const Def* normalize_ICmp(const Def*, const Def*, const Def*) { return nullptr; }

template<RCmp op>
const Def* normalize_RCmp(const Def*, const Def*, const Def*) { return nullptr; }

template<_Cast op>
const Def* normalize__Cast(const Def*, const Def*, const Def*) { return nullptr; }

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
