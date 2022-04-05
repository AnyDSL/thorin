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

const Def* Importer::import(Tracker odef) {
    if (auto ndef = def_old2new_.lookup(odef)) {
        assert(&(*ndef)->world() == &world_);
        assert(!(*ndef)->is_replaced());
        return *ndef;
    }

    auto ntype = import(odef->type());

    if (auto oparam = odef->isa<Param>()) {
        auto ncont = import(oparam->continuation())->as_continuation();
        auto nparam = ncont->param(oparam->index());
        assert(nparam && &nparam->world() == &world_);
        assert(!nparam->is_replaced());
        return nparam;
    }

    Continuation* ncontinuation = nullptr;
    if (auto ocontinuation = odef->isa_continuation()) { // create stub in new world
        // TODO maybe we want to deal with intrinsics in a more streamlined way
        if (ocontinuation == ocontinuation->world().branch())
            return def_old2new_[ocontinuation] = world().branch();
        if (ocontinuation == ocontinuation->world().end_scope())
            return def_old2new_[ocontinuation] = world().end_scope();
        auto npi = import(ocontinuation->type())->as<FnType>();
        ncontinuation = world().continuation(npi, ocontinuation->attributes(), ocontinuation->debug_history());
        assert(&ncontinuation->world() == &world());
        assert(&npi->table() == &world());
        for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
            ncontinuation->param(i)->set_name(ocontinuation->param(i)->debug_history().name);
            def_old2new_[ocontinuation->param(i)] = ncontinuation->param(i);
        }

        def_old2new_[ocontinuation] = ncontinuation;

        if (ocontinuation->num_ops() > 0 && ocontinuation->callee() == ocontinuation->world().branch()) {
            auto cond = import(ocontinuation->arg(1));
            if (auto lit = cond->isa<PrimLit>()) {
                auto callee = import(lit->value().get_bool() ? ocontinuation->arg(2) : ocontinuation->arg(3));
                auto mem = import(ocontinuation->arg(0));
                ncontinuation->jump(callee, {mem}, ocontinuation->debug()); // TODO debug

                assert(!ncontinuation->is_replaced());
                return ncontinuation;
            }
        }

        auto old_profile = ocontinuation->filter();
        Array<const Def*> new_profile(old_profile.size());
        for (size_t i = 0, e = old_profile.size(); i != e; ++i)
            new_profile[i] = import(old_profile[i]);
        ncontinuation->set_filter(new_profile);
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(odef->op(i));
        assert(&nops[i]->world() == &world());
    }

    if (auto oprimop = odef->isa<PrimOp>()) {
        auto nprimop = oprimop->rebuild(world(), ntype, nops);
        todo_ |= oprimop->tag() != nprimop->tag();
        assert(!nprimop->is_replaced());
        return def_old2new_[oprimop] = nprimop;
    }

    auto ocontinuation = odef->as_continuation();
    assert(ncontinuation && &ncontinuation->world() == &world());
    if (size > 0)
        ncontinuation->jump(nops.front(), nops.skip_front(), ocontinuation->debug()); // TODO debug
    assert(!ncontinuation->is_replaced());
    return ncontinuation;
}

}
