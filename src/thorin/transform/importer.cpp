#include "thorin/transform/importer.h"

namespace thorin {

Importer::Importer(World& src)
    : world_(src)
{
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
    if (auto onom = odef->isa_nominal()) {
        ndef = onom->stub(world_, ntype);
        old2new_[odef] = ndef;
    }

    auto num_ops = odef->num_ops();
    Array<const Def*> nops(num_ops, [&](size_t i) { return import(odef->op(i)); });

    if (ndef) {
        for (size_t i = 0; i != num_ops; ++i)
            ndef->as_nominal()->set(i, nops[i]);
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
