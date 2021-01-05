#include "thorin/axiom.h"

namespace thorin {

Axiom::Axiom(NormalizeFn normalizer, const Def* type, u32 tag, u32 flags, const Def* dbg)
    : Def(Node, type, Defs{}, (nat_t(tag) << 32_u64) | nat_t(flags), dbg)
{
    u16 currying_depth = 0;
    while (auto pi = type->isa<Pi>()) {
        ++currying_depth;
        type = pi->codom();
    }

    normalizer_depth_.set(normalizer, currying_depth);
}

std::tuple<const Axiom*, u16> get_axiom(const Def* def) {
    if (auto axiom = def->isa<Axiom>()) return {axiom, axiom->currying_depth()};
    if (auto app = def->isa<App>()) return {app->axiom(), app->currying_depth()};
    return {0, u16(-1)};
}

bool is_memop(const Def* def) { return def->isa<App>() && isa<Tag::Mem>(def->out(0)->type()); }

}
