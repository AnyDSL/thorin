#include "thorin/world.h"

namespace thorin {

Type import(Type2Type& old2new, World& to, Type otype) {
    if (auto ntype = find(old2new, otype)) {
        assert(&ntype->world() == &to);
        return ntype;
    }

    size_t size = otype->size();
    Array<Type> nelems(size);
    for (size_t i = 0; i != size; ++i)
        nelems[i] = import(old2new, to, otype->elem(i));
    
    auto ntype = old2new[otype] = World::rebuild(to, otype, nelems);
    assert(&ntype->world() == &to);
    return ntype;
}

Def import(Type2Type& type_old2new, Def2Def& def_old2new, World& to, Def odef) {
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
        auto npi = import(type_old2new, to, olambda->pi())->as<Pi>();
        nlambda = to.lambda(npi, olambda->attribute(), olambda->name);
        for (size_t i = 0, e = olambda->num_params(); i != e; ++i) {
            nlambda->param(i)->name = olambda->param(i)->name;
            def_old2new[olambda->param(i)] = nlambda->param(i);
        }

        def_old2new[olambda] = nlambda;
    }

    size_t size = odef->size();
    Array<Def> nops(size);
    for (size_t i = 0; i != size; ++i) {
        nops[i] = import(type_old2new, def_old2new, to, odef->op(i));
        assert(&nops[i]->world() == &to);
    }
    
    if (auto oprimop = odef->isa<PrimOp>())
        return def_old2new[oprimop] = World::rebuild(to, oprimop, nops, ntype);

    assert(nlambda && &nlambda->world() == &to);
    if (size > 0)
        nlambda->jump(nops[0], nops.slice_from_begin(1));
    return nlambda;
}

Type import(World& to, Type otype) {
    Type2Type old2new;
    return import(old2new, to, otype);
}

Def import(World& to, Def odef) {
    Def2Def def_old2new;
    Type2Type type_old2new;
    return import(type_old2new, def_old2new, to, odef);
}

}
