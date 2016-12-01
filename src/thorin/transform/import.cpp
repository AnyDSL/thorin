#include "thorin/world.h"

namespace thorin {

const Type* import(World& to, Type2Type& old2new, const Type* otype) {
    if (auto ntype = find(old2new, otype)) {
        assert(&ntype->world() == &to);
        return ntype;
    }
    size_t size = otype->num_ops();

    if (auto struct_type = otype->isa<StructType>()) {
        auto ntype = to.struct_type(struct_type->name(), struct_type->num_ops());
        old2new[otype] = ntype;
        for (size_t i = 0; i != size; ++i)
            ntype->set(i, import(to, old2new, otype->op(i)));
        return ntype;
    }

    Array<const Type*> nops(size);
    for (size_t i = 0; i != size; ++i)
        nops[i] = import(to, old2new, otype->op(i));

    auto ntype = otype->rebuild(to, nops);
    old2new[otype] = ntype;
    assert(&ntype->world() == &to);

    return ntype;
}

const Def* import(World& to, Type2Type& type_old2new, Def2Def& def_old2new, const Def* odef) {
    if (auto ndef = find(def_old2new, odef)) {
        assert(&ndef->world() == &to);
        return ndef;
    }

    auto ntype = import(to, type_old2new, odef->type());

    if (auto oparam = odef->isa<Param>()) {
        import(to, type_old2new, def_old2new, oparam->continuation())->as_continuation();
        auto nparam = find(def_old2new, oparam);
        assert(nparam && &nparam->world() == &to);
        return nparam;
    }

    Continuation* ncontinuation = nullptr;
    if (auto ocontinuation = odef->isa_continuation()) { // create stub in new world
        // TODO maybe we want to deal with intrinsics in a more streamlined way
        if (ocontinuation == ocontinuation->world().branch())
            return def_old2new[ocontinuation] = to.branch();
        if (ocontinuation == ocontinuation->world().end_scope())
            return def_old2new[ocontinuation] = to.end_scope();
        auto npi = import(to, type_old2new, ocontinuation->type())->as<FnType>();
        ncontinuation = to.continuation(npi, ocontinuation->loc(), ocontinuation->cc(), ocontinuation->intrinsic(), ocontinuation->name);
        assert(&ncontinuation->world() == &to);
        assert(&npi->world() == &to);
        for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
            ncontinuation->param(i)->name = ocontinuation->param(i)->name;
            def_old2new[ocontinuation->param(i)] = ncontinuation->param(i);
        }

        def_old2new[ocontinuation] = ncontinuation;

        if (ocontinuation->is_external())
            ncontinuation->make_external();

        if (ocontinuation->num_ops() > 0 && ocontinuation->callee() == ocontinuation->world().branch()) {
            auto cond = import(to, type_old2new, def_old2new, ocontinuation->arg(0));
            if (auto lit = cond->isa<PrimLit>()) {
                auto callee = import(to, type_old2new, def_old2new, lit->value().get_bool() ? ocontinuation->arg(1) : ocontinuation->arg(2));
                ncontinuation->jump(callee, {}, ocontinuation->jump_loc());
                return ncontinuation;
            }
        }
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(to, type_old2new, def_old2new, odef->op(i));
        assert(&nops[i]->world() == &to);
    }

    if (auto oprimop = odef->isa<PrimOp>()) {
        auto nprimop = oprimop->rebuild(to, nops, ntype);
        return def_old2new[oprimop] = nprimop;
    }

    auto ocontinuation = odef->as_continuation();
    assert(ncontinuation && &ncontinuation->world() == &to);
    if (size > 0)
        ncontinuation->jump(nops.front(), nops.skip_front(), ocontinuation->jump_loc());
    return ncontinuation;
}

const Def* import(World& to, const Def* odef) {
    Def2Def def_old2new;
    Type2Type type_old2new;
    return import(to, type_old2new, def_old2new, odef);
}

}
