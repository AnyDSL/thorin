#include "thorin/world.h"

namespace thorin {

const Type* import(Type2Type& old2new, World& to, const Type* otype) {
    if (auto ntype = find(old2new, otype)) {
        assert(&ntype->world() == &to);
        return ntype;
    }

    size_t size = otype->num_args();
    Array<const Type*> nargs(size);
    for (size_t i = 0; i != size; ++i)
        nargs[i] = import(old2new, to, otype->arg(i));

    auto ntype = old2new[otype] = otype->rebuild(to, nargs);
    assert(&ntype->world() == &to);

    Array<const TypeParam*> ntype_params(otype->num_type_params());
    for (size_t i = 0, e = otype->num_type_params(); i != e; ++i)
        ntype_params[i] = import(old2new, to, otype->type_param(i))->as<TypeParam>();

    return close(ntype, ntype_params);
}

const Def* import(Type2Type& type_old2new, Def2Def& def_old2new, World& to, const Def* odef) {
    if (auto ndef = find(def_old2new, odef)) {
        assert(&ndef->world() == &to);
        return ndef;
    }

    auto ntype = import(type_old2new, to, odef->type());

    if (auto oparam = odef->isa<Param>()) {
        import(type_old2new, def_old2new, to, oparam->continuation())->as_continuation();
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
        auto npi = import(type_old2new, to, ocontinuation->type())->as<FnType>();
        ncontinuation = to.continuation(npi, ocontinuation->loc(), ocontinuation->cc(), ocontinuation->intrinsic(), ocontinuation->name);
        for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
            ncontinuation->param(i)->name = ocontinuation->param(i)->name;
            def_old2new[ocontinuation->param(i)] = ncontinuation->param(i);
        }

        def_old2new[ocontinuation] = ncontinuation;
    }

    size_t size = odef->size();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(type_old2new, def_old2new, to, odef->op(i));
        assert(&nops[i]->world() == &to);
    }

    if (auto oprimop = odef->isa<PrimOp>())
        return def_old2new[oprimop] = oprimop->rebuild(to, nops, ntype);

    auto ocontinuation = odef->as_continuation();
    Array<const Type*> ntype_args(ocontinuation->type_args().size());
    for (size_t i = 0, e = ntype_args.size(); i != e; ++i)
        ntype_args[i] = import(type_old2new, to, ocontinuation->type_arg(i));

    assert(ncontinuation && &ncontinuation->world() == &to);
    if (size > 0)
        ncontinuation->jump(nops.front(), ntype_args, nops.skip_front(), ocontinuation->jump_loc());
    return ncontinuation;
}

const Type* import(World& to, const Type* otype) {
    Type2Type old2new;
    return import(old2new, to, otype);
}

const Def* import(World& to, const Def* odef) {
    Def2Def def_old2new;
    Type2Type type_old2new;
    return import(type_old2new, def_old2new, to, odef);
}

}
