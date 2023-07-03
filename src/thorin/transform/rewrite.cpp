#include "rewrite.h"

namespace thorin {

Rewriter::Rewriter(World& src, World& dst) : src_(src), dst_(dst) {
    old2new_.rehash(src.defs().capacity());
}

Rewriter::Rewriter(World& src, World& dst, Rewriter& parent) : Rewriter(src, dst) {
    old2new_ = parent.old2new_;
}

const Def* Rewriter::lookup(const thorin::Def* odef) {
    if (auto ndef = old2new_.lookup(odef)) return *ndef;

    // TODO maybe we want to deal with intrinsics in a more streamlined way
    if (odef == src().branch())
        return dst().branch();
    if (odef == src().end_scope())
        return dst().end_scope();
    return nullptr;
}

const Def* Rewriter::instantiate(const Def* odef) {
    auto found = lookup(odef);
    if (found) return found;

    return old2new_[odef] = rewrite(odef);
}

const Def* Rewriter::insert(const Def* odef, const Def* ndef) {
    assert(&odef->world() == &src());
    assert(&ndef->world() == &dst());
    return old2new_[odef] = ndef;
}

const Def* Rewriter::rewrite(const Def* odef) {
    if (odef == odef->world().star())
        return insert(odef, dst().star());

    auto ntype = instantiate(odef->type())->as<Type>();

    Def* stub = nullptr;
    if (odef->isa_nom()) {
        stub = odef->stub(*this, ntype);
        insert(odef, stub);
    }

    if (odef->isa_structural()) {
        size_t size = odef->num_ops();
        Array<const Def*> nops(size);
        for (size_t i = 0; i != size; ++i) {
            assert(odef->op(i) != odef);
            nops[i] = instantiate(odef->op(i));
            assert(&nops[i]->world() == &dst());
        }
        auto ndef = odef->rebuild(dst(), ntype, nops);
        return ndef;
    } else {
        assert(odef->isa_nom() && stub);
        stub->rebuild_from(*this, odef);
        return stub;
    }
}

}