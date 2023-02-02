#include "rewrite.h"

namespace thorin {

Rewriter::Rewriter(World& src, World& dst) : src_(src), dst_(dst) {
    old2new_.rehash(src.defs().capacity());
}

const Def* Rewriter::instantiate(const Def* odef) {
    if (auto ndef = old2new_.lookup(odef)) return *ndef;

    return old2new_[odef] = rewrite(odef);
}

const Def* Rewriter::insert(const Def* odef, const Def* ndef) {
    return old2new_[odef] = ndef;
}

const Def* Rewriter::rewrite(const Def* odef) {
    if (odef == odef->world().star())
        return insert(odef, dst().star());

    auto ntype = instantiate(odef->type())->as<Type>();

    Def* stub = nullptr;
    if (odef->isa_nom()) {
        stub = odef->stub(dst(), ntype);
        insert(odef, stub);
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        assert(odef->op(i) != odef);
        nops[i] = instantiate(odef->op(i));
        assert(&nops[i]->world() == &dst());
    }

    if (odef->isa_structural()) {
        auto ndef = odef->rebuild(dst(), ntype, nops);
        return ndef;
    } else {
        assert(odef->isa_nom() && stub);
        stub->rebuild_from(odef, nops);
        return stub;
    }
}

}