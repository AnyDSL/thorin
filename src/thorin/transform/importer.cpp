#include "thorin/transform/importer.h"

namespace thorin {

Importer::Importer(World& src)
    : world_(src)
{
    old2new_[src.branch()]    = world().branch();
    old2new_[src.end_scope()] = world().end_scope();
    old2new_[src.universe()]  = world().universe();
}

const Def* Importer::import(Tracker odef) {
    if (auto ndef = old2new_.lookup(odef)) {
        assert(!(*ndef)->is_replaced());
        return *ndef;
    }

    auto ntype = import(odef->type());

    const Def* ndef = nullptr;
    if (odef->isa_nominal()) {
        ndef = odef->stub(world_, ntype);
        old2new_[odef] = ndef;
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i)
        nops[i] = import(odef->op(i));

    if (ndef) {
        for (size_t i = 0; i != size; ++i)
            const_cast<Def*>(ndef)->set(i, nops[i]);
        if (auto olam = odef->isa<Lam>()) { // TODO do sth smarter here
            if (olam->is_external())
                ndef->as_nominal<Lam>()->make_external();
        }
    } else {
        ndef = odef->rebuild(world_, ntype, nops);
        old2new_[odef] = ndef;
    }

    assert(&ndef->world() == &world());
    return ndef;
}

}
