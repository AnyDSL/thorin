#include "thorin/transform/importer.h"

namespace thorin {

const Type* Importer::import(const Type* otype) {
    if (auto ntype = type_old2new_.lookup(otype)) {
        assert(&(*ntype)->table() == &world_);
        return *ntype;
    }
    size_t size = otype->num_ops();

    if (auto nominal_type = otype->isa<NominalType>()) {
        auto ntype = nominal_type->stub(world_);
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

const Def* Importer::import(const Def* odef) {
    if (auto ndef = def_old2new_.lookup(odef)) {
        assert(&(*ndef)->world() == &world_);
        return *ndef;
    }

    auto ntype = import(odef->type());

    if (auto oparam = odef->isa<Param>()) {
        import(oparam->continuation())->as_nom<Lam>();
        auto nparam = def_old2new_[oparam];
        assert(nparam && &nparam->world() == &world_);
        return nparam;
    }

    if (auto ofilter = odef->isa<Filter>()) {
        Array<const Def*> new_conditions(ofilter->num_ops());
        for (size_t i = 0, e = ofilter->size(); i != e; ++i)
            new_conditions[i] = import(ofilter->condition(i));
        auto nfilter = world().filter(new_conditions, ofilter->debug());
        return nfilter;
    }

    Lam* ncontinuation = nullptr;
    if (auto ocontinuation = odef->isa_nom<Lam>()) { // create stub in new world
        assert(!ocontinuation->dead_);
        // TODO maybe we want to deal with intrinsics in a more streamlined way
        if (ocontinuation == ocontinuation->world().branch())
            return def_old2new_[ocontinuation] = world().branch();
        if (ocontinuation == ocontinuation->world().end_scope())
            return def_old2new_[ocontinuation] = world().end_scope();
        auto npi = import(ocontinuation->type())->as<FnType>();
        ncontinuation = world().lambda(npi, ocontinuation->attributes(), ocontinuation->debug_history());
        assert(&ncontinuation->world() == &world());
        assert(&npi->table() == &world());
        for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
            ncontinuation->param(i)->set_name(ocontinuation->param(i)->debug_history().name);
            def_old2new_[ocontinuation->param(i)] = ncontinuation->param(i);
        }

        def_old2new_[ocontinuation] = ncontinuation;

        if (ocontinuation->is_external())
            world().make_external(ncontinuation);
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
    }

    assert(ncontinuation && &ncontinuation->world() == &world());
    auto napp = nops[0]->isa<App>();
    if (napp)
        ncontinuation->set_body(napp);
    ncontinuation->set_filter(nops[1]->as<Filter>());
    ncontinuation->verify();
    return ncontinuation;
}

}
