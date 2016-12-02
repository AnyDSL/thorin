#include "thorin/transform/importer.h"

namespace thorin {

const Type* Importer::import(const Type* otype) {
    if (auto ntype = find(type_old2new_, otype)) {
        assert(&ntype->world() == &world_);
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
    assert(&ntype->world() == &world_);

    return ntype;
}

const Def* Importer::import(const Def* odef) {
    if (auto ndef = find(def_old2new_, odef)) {
        assert(&ndef->world() == &world_);
        return ndef;
    }

    auto ntype = import(odef->type());

    if (auto oparam = odef->isa<Param>()) {
        import(oparam->continuation())->as_continuation();
        auto nparam = find(def_old2new_, oparam);
        assert(nparam && &nparam->world() == &world_);
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
        ncontinuation = world().continuation(npi, ocontinuation->cc(), ocontinuation->intrinsic(), ocontinuation->debug());
        assert(&ncontinuation->world() == &world());
        assert(&npi->world() == &world());
        for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
            ncontinuation->param(i)->debug().set(ocontinuation->param(i)->name());
            def_old2new_[ocontinuation->param(i)] = ncontinuation->param(i);
        }

        def_old2new_[ocontinuation] = ncontinuation;

        if (ocontinuation->is_external())
            ncontinuation->make_external();

        if (ocontinuation->num_ops() > 0 && ocontinuation->callee() == ocontinuation->world().branch()) {
            auto cond = import(ocontinuation->arg(0));
            if (auto lit = cond->isa<PrimLit>()) {
                auto callee = import(lit->value().get_bool() ? ocontinuation->arg(1) : ocontinuation->arg(2));
                ncontinuation->jump(callee, {}, ocontinuation->jump_debug());
                return ncontinuation;
            }
        }
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(odef->op(i));
        assert(&nops[i]->world() == &world());
    }

    if (auto oprimop = odef->isa<PrimOp>()) {
        auto nprimop = oprimop->rebuild(world(), nops, ntype);
        return def_old2new_[oprimop] = nprimop;
    }

    auto ocontinuation = odef->as_continuation();
    assert(ncontinuation && &ncontinuation->world() == &world());
    if (size > 0)
        ncontinuation->jump(nops.front(), nops.skip_front(), ocontinuation->jump_debug());
    return ncontinuation;
}

}
