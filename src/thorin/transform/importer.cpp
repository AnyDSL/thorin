#include "thorin/transform/importer.h"

namespace thorin {

Importer::Importer(World& src)
    : world_(src.name())
{
    if  (src.is_pe_done())
        world_.mark_pe_done();
#if THORIN_ENABLE_CHECKS
    if (src.track_history())
        world_.enable_history(true);
#endif

    old2new_[src.branch()]    = world().branch();
    old2new_[src.end_scope()] = world().end_scope();
    old2new_[src.universe()]  = world().universe();
}

const Def* Importer::import(Tracker odef) {
    if (auto ndef = find(old2new_, odef)) {
        assert(&ndef->world() == &world_);
        assert(!ndef->is_replaced());
        return ndef;
    }

    auto ntype = import(odef->type());

    const Def* ndef = nullptr;
    if (odef->is_nominal()) {
        ndef = odef->stub(world_, ntype);
        old2new_[odef] = ndef;
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(odef->op(i));
        assert(&nops[i]->world() == &world());
    }

    if (ndef) {
        for (size_t i = 0; i != size; ++i)
            const_cast<Def*>(ndef)->update(i, nops[i]); // TODO use set_op here
        if (auto olam = odef->isa<Lam>()) { // TODO do sth smarter here
            if (olam->is_external())
                ndef->as_lam()->make_external();
        }
    } else {
        ndef = odef->rebuild(world_, ntype, nops);
        old2new_[odef] = ndef;
    }

    assert(&ndef->world() == &world());
    return ndef;
}

}
