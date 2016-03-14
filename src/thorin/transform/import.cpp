#include "thorin/world.h"

namespace thorin {

const Type* import(Type2Type& old2new, World& to, const Type* otype) {
    if (auto ntype = find(old2new, otype)) {
        assert(&ntype->world() == &to);
        return ntype;
    }

    size_t size = otype->size();
    Array<const Type*> nargs(size);
    for (size_t i = 0; i != size; ++i)
        nargs[i] = import(old2new, to, otype->arg(i));

    auto ntype = old2new[otype] = otype->rebuild(to, nargs);
    assert(&ntype->world() == &to);

    //return close(ntype, ntype_params);
    return ntype;
}

const Def* import(Type2Type& type_old2new, Def2Def& def_old2new, World& to, const Def* odef) {
    if (auto ndef = find(def_old2new, odef)) {
        assert(&ndef->world() == &to);
        return ndef;
    }

    auto ntype = import(type_old2new, to, odef->type());

    if (auto oparam = odef->isa<Param>()) {
        import(type_old2new, def_old2new, to, oparam->lambda())->as_lambda();
        auto nparam = find(def_old2new, oparam);
        assert(nparam && &nparam->world() == &to);
        return nparam;
    }

    Lambda* nlambda = nullptr;
    if (auto olambda = odef->isa_lambda()) { // create stub in new world
        // TODO maybe we want to deal with intrinsics in a more streamlined way
        if (olambda == olambda->world().branch())
            return def_old2new[olambda] = to.branch();
        if (olambda == olambda->world().end_scope())
            return def_old2new[olambda] = to.end_scope();
        auto npi = import(type_old2new, to, olambda->type())->as<FnType>();
        nlambda = to.lambda(npi, olambda->loc(), olambda->cc(), olambda->intrinsic(), olambda->name);
        for (size_t i = 0, e = olambda->num_params(); i != e; ++i) {
            nlambda->param(i)->name = olambda->param(i)->name;
            def_old2new[olambda->param(i)] = nlambda->param(i);
        }

        def_old2new[olambda] = nlambda;
    }

    size_t size = odef->size();
    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(type_old2new, def_old2new, to, odef->op(i));
        assert(&nops[i]->world() == &to);
    }

    if (auto oprimop = odef->isa<PrimOp>())
        return def_old2new[oprimop] = oprimop->rebuild(to, nops, ntype);

    auto olambda = odef->as_lambda();
    Array<const Type*> ntype_args(olambda->type_args().size());
    for (size_t i = 0, e = ntype_args.size(); i != e; ++i)
        ntype_args[i] = import(type_old2new, to, olambda->type_arg(i));

    assert(nlambda && &nlambda->world() == &to);
    if (size > 0)
        nlambda->jump(nops.front(), ntype_args, nops.skip_front(), olambda->jump_loc());
    return nlambda;
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
