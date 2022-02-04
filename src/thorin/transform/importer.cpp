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
        import(oparam->lambda())->as_nom<Lam>();
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

    Lam* nlam = nullptr;
    if (auto olam = odef->isa_nom<Lam>()) { // create stub in new world
        assert(!olam->dead_);
        // TODO maybe we want to deal with intrinsics in a more streamlined way
        if (olam == olam->world().branch())
            return def_old2new_[olam] = world().branch();
        if (olam == olam->world().end_scope())
            return def_old2new_[olam] = world().end_scope();
        auto npi = import(olam->type())->as<FnType>();
        nlam = world().lambda(npi, olam->attributes(), olam->debug_history());
        assert(&nlam->world() == &world());
        assert(&npi->table() == &world());
        for (size_t i = 0, e = olam->num_params(); i != e; ++i) {
            nlam->param(i)->set_name(olam->param(i)->debug_history().name);
            def_old2new_[olam->param(i)] = nlam->param(i);
        }

        def_old2new_[olam] = nlam;

        if (olam->is_external())
            world().make_external(nlam);
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

    assert(nlam && &nlam->world() == &world());
    auto napp = nops[0]->isa<App>();
    if (napp)
        nlam->set_body(napp);
    nlam->set_filter(nops[1]->as<Filter>());
    nlam->verify();
    return nlam;
}

}
