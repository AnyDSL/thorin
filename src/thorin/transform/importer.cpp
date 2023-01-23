#include "thorin/transform/importer.h"

namespace thorin {

const Def* Importer::import(const Def* odef) {
    if (auto ndef = def_old2new_.lookup(odef)) {
        assert(&(*ndef)->world() == &world());
        return *ndef;
    }

    if (odef == odef->world().star()) {
        def_old2new_[odef] = world().star();
        return world().star();
    }

    auto ntype = import(odef->type())->as<Type>();

    Def* stub = nullptr;
    if (odef->isa_nom()) {
        stub = odef->stub(world(), ntype);
        def_old2new_[odef] = stub;
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        assert(odef->op(i) != odef);
        nops[i] = import(odef->op(i));
        assert(&nops[i]->world() == &world());
    }

    if (odef->isa_structural()) {
        auto ndef = odef->rebuild(world(), ntype, nops);
        todo_ |= odef->tag() != ndef->tag();
        return def_old2new_[odef] = ndef;
    } else {
        assert(odef->isa_nom() && stub);
        stub->rebuild_from(odef, nops);
        return stub;
    }
}

}
