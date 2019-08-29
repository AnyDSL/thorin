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

    return world.app(callee, {cond, a, b}, dbg);
}


}
