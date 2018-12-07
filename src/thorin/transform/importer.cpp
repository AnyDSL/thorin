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

    def_old2new_[src.branch()]    = world().branch();
    def_old2new_[src.end_scope()] = world().end_scope();
}

const Type* Importer::import(const Type* otype) {
    if (auto ntype = find(type_old2new_, otype)) {
        assert(&ntype->table() == &world_);
        return ntype;
    }

    size_t size = otype->num_ops();

    if (auto struct_type = otype->isa<StructType>()) {
        auto ntype = world_.struct_type(struct_type->name(), struct_type->num_ops());
        type_old2new_[otype] = ntype;
        for (size_t i = 0; i != size; ++i)
            ntype->set(i, import(otype->op(i)));
        return ntype;
    }

    Array<const Type*> nops(size);
    for (size_t i = 0; i != size; ++i)
        nops[i] = import(otype->op(i));

    auto ntype = otype->rebuild(world_, nops);
    type_old2new_[otype] = ntype;
    assert(&ntype->table() == &world_);

    return ntype;
}

const Def* Importer::import(Tracker odef) {
    if (auto ndef = find(def_old2new_, odef)) {
        assert(&ndef->world() == &world_);
        assert(!ndef->is_replaced());
        return ndef;
    }

    auto ntype = import(odef->type());

    const Def* ndef = nullptr;
    if (odef->is_nominal()) {
        ndef = odef->vstub(world_, ntype);
        def_old2new_[odef] = ndef;
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(odef->op(i));
        assert(&nops[i]->world() == &world());
    }

    if (ndef) {
        for (size_t i = 0; i != size; ++i)
            const_cast<Def*>(ndef)->update_op(i, nops[i]); // TODO use set_op here
    } else {
        ndef = odef->vrebuild(world_, ntype, nops);
        def_old2new_[odef] = ndef;
    }

    assert(&ndef->world() == &world());
    return ndef;
}

}
